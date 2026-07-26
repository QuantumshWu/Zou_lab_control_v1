"""Focused lifecycle contracts for live faceted panel composition."""

from __future__ import annotations

from dataclasses import replace

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
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.display_range import RelimMode
from zlc_frontend.data_figure import DataFigure
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    FigureEvaluator,
    FixedIndex,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    suggest_view,
)
from zlc_frontend.histogram_display import (
    FacetedHistogramDisplayState,
    HistogramCountScale,
    HistogramDisplayState,
)
from zlc_frontend.image_display import ImageDisplayState
from zlc_frontend.meter_display import MeterDisplayState
from zlc_frontend.panel_render import (
    FacetedPanelFocus,
    PanelComposer,
    PanelProvenance,
)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _curve_source(revision: int):
    repeat = _axis("faceted-lifecycle.curve.repeat", REPEAT, 2)
    point = _axis("faceted-lifecycle.curve.point", SCAN_POINT, 4)
    site = _axis("faceted-lifecycle.curve.site", SITE, 2)
    component = _axis("faceted-lifecycle.curve.component", COMPONENT, 2)
    values = (
        np.arange(32, dtype=np.float64).reshape(2, 4, 2, 2)
        + revision * 100.0
    )
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((point.size,)),
        ValueSchema(
            (site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("faceted-lifecycle-curve"),
        DatasetRevision(revision),
        values,
        DatasetComponentValidity(
            (site.axis_id, component.axis_id),
            np.ones(values.shape, dtype=bool),
        ),
        schema,
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(facet_axis_ids=(site.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    return (
        OwnedSnapshot(
            block.ref(StreamGenerationId("faceted-lifecycle-curve-generation")),
            block,
        ),
        suggestion.spec,
        CurveDisplayState(),
    )


def _histogram_source(revision: int):
    repeat = _axis("faceted-lifecycle.histogram.repeat", REPEAT, 4)
    point = _axis("faceted-lifecycle.histogram.point", SCAN_POINT, 1)
    site = _axis("faceted-lifecycle.histogram.site", SITE, 2)
    values = (
        np.arange(8, dtype=np.float64).reshape(4, 1, 2)
        + revision * 100.0
    )
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((point.size,)),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("faceted-lifecycle-histogram"),
        DatasetRevision(revision),
        values,
        DatasetComponentValidity(
            (site.axis_id,),
            np.ones(values.shape, dtype=bool),
        ),
        schema,
    )
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
            AxisViewBinding(site.axis_id, AxisViewRole.FACET),
        ),
    )
    return (
        OwnedSnapshot(
            block.ref(
                StreamGenerationId("faceted-lifecycle-histogram-generation")
            ),
            block,
        ),
        view,
        FacetedHistogramDisplayState(HistogramDisplayState()),
    )


def _image_source(revision: int):
    repeat = _axis("faceted-lifecycle.image.repeat", REPEAT, 1)
    x = _axis("faceted-lifecycle.image.x", SCAN_POINT, 3)
    y = _axis("faceted-lifecycle.image.y", SCAN_POINT, 2)
    facet = _axis("faceted-lifecycle.image.facet", SCAN_POINT, 2)
    values = (
        np.arange(12, dtype=np.float64).reshape(1, 12, 1)
        + revision * 100.0
    )
    schema = DatasetSchema(
        repeat,
        (x, y, facet),
        PointLayout.rect_c((x.size, y.size, facet.size)),
        ValueSchema.scalar(values.dtype, "count"),
    )
    block = DataBlock(
        BlockId("faceted-lifecycle-image"),
        DatasetRevision(revision),
        values,
        VALID,
        schema,
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.IMAGE,
        preferences=ViewPreferences(
            image_x_axis_id=x.axis_id,
            image_y_axis_id=y.axis_id,
            facet_axis_ids=(facet.axis_id,),
        ),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    return (
        OwnedSnapshot(
            block.ref(StreamGenerationId("faceted-lifecycle-image-generation")),
            block,
        ),
        suggestion.spec,
        ImageDisplayState(),
    )


def _meter_source(revision: int):
    repeat = _axis("faceted-lifecycle.meter.repeat", REPEAT, 2)
    site = _axis("faceted-lifecycle.meter.site", SITE, 2)
    values = (
        np.arange(4, dtype=np.float64).reshape(2, 1, 2)
        + revision * 100.0
    )
    schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("faceted-lifecycle-meter"),
        DatasetRevision(revision),
        values,
        DatasetComponentValidity(
            (site.axis_id,),
            np.ones(values.shape, dtype=bool),
        ),
        schema,
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.METER,
        preferences=ViewPreferences(facet_axis_ids=(site.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    return (
        OwnedSnapshot(
            block.ref(StreamGenerationId("faceted-lifecycle-meter-generation")),
            block,
        ),
        suggestion.spec,
        MeterDisplayState(0, None),
    )


@pytest.mark.parametrize(
    ("intent", "source_factory"),
    (
        (ViewIntent.CURVE, _curve_source),
        (ViewIntent.HISTOGRAM, _histogram_source),
    ),
)
def test_faceted_display_reuses_one_evaluation_focus_and_agg_surface(
    monkeypatch,
    intent,
    source_factory,
) -> None:
    snapshot, view, display = source_factory(1)
    next_snapshot, _next_view, _next_display = source_factory(2)
    provenance = PanelProvenance("run", "epoch", "0" * 64)
    evaluate_calls = 0
    focus_calls = 0
    original_evaluate = FigureEvaluator.evaluate
    original_focus = DataFigure.focused_typed_panel

    def counted_evaluate(self, *args, **kwargs):
        nonlocal evaluate_calls
        evaluate_calls += 1
        return original_evaluate(self, *args, **kwargs)

    def counted_focus(self, *args, **kwargs):
        nonlocal focus_calls
        focus_calls += 1
        return original_focus(self, *args, **kwargs)

    monkeypatch.setattr(FigureEvaluator, "evaluate", counted_evaluate)
    monkeypatch.setattr(DataFigure, "focused_typed_panel", counted_focus)
    composer = PanelComposer(
        f"faceted-lifecycle-{intent.value.lower()}",
        intent=intent,
        size_name="1x2",
        view=view,
    )
    try:
        overview = composer.compose_faceted(
            snapshot,
            display=display,
            provenance=provenance,
        )
        overview_renderer = composer._faceted_overview_renderer
        assert overview_renderer is not None
        overview_surface = overview_renderer._figure
        assert overview_surface is not None
        overview_axes = tuple(id(axis) for axis in overview_renderer._axes)
        overview_artists = tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in overview_renderer._cells
        )
        repeated_overview = composer.compose_faceted(
            snapshot,
            display=replace(display, display=replace(display.display, revision=1))
            if isinstance(display, FacetedHistogramDisplayState)
            else replace(display, revision=1),
            provenance=provenance,
        )
        assert repeated_overview.figure is overview.figure
        assert composer._faceted_overview_renderer is overview_renderer
        assert overview_renderer._figure is overview_surface
        assert tuple(id(axis) for axis in overview_renderer._axes) == overview_axes
        assert tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in overview_renderer._cells
        ) == overview_artists
        assert evaluate_calls == 1

        assert overview.overview is not None
        region = overview.overview.regions[0]
        focus = FacetedPanelFocus(0, region.focus_selection)
        first_focus = composer.compose_faceted(
            snapshot,
            display=display,
            provenance=provenance,
            focus=focus,
        )
        renderer = composer._faceted_renderer
        focused_figure = composer._faceted_focus_figure
        assert renderer is not None
        assert focused_figure is not None
        artist_ids = tuple(id(artist) for artist in renderer._artists)
        assert artist_ids
        blit_cache = renderer._blit_cache

        revised_display = (
            replace(display, display=replace(display.display, revision=2))
            if isinstance(display, FacetedHistogramDisplayState)
            else replace(display, revision=2)
        )
        second_focus = composer.compose_faceted(
            snapshot,
            display=revised_display,
            provenance=provenance,
            focus=focus,
        )
        assert second_focus.figure is first_focus.figure is overview.figure
        assert composer._faceted_focus_figure is focused_figure
        assert composer._faceted_renderer is renderer
        assert tuple(id(artist) for artist in renderer._artists) == artist_ids
        assert renderer._blit_cache is blit_cache
        assert evaluate_calls == 1
        assert focus_calls == 1

        other_region = overview.overview.regions[1]
        other_focus = FacetedPanelFocus(1, other_region.focus_selection)
        composer.compose_faceted(
            snapshot,
            display=revised_display,
            provenance=provenance,
            focus=other_focus,
        )
        next_renderer = composer._faceted_renderer
        next_focused_figure = composer._faceted_focus_figure
        assert next_renderer is not None and next_renderer is not renderer
        assert next_focused_figure is not None
        assert renderer._figure is None
        assert evaluate_calls == 1
        assert focus_calls == 2

        revised_focus = composer.compose_faceted(
            next_snapshot,
            display=revised_display,
            provenance=provenance,
            focus=other_focus,
        )
        assert evaluate_calls == 2
        assert focus_calls == 3
        assert composer._faceted_renderer is next_renderer
        assert next_renderer._figure is not None
        assert composer._faceted_focus_figure is not next_focused_figure
        assert (
            composer._faceted_focus_figure.document.document_id
            == next_focused_figure.document.document_id
        )
        assert (
            revised_focus.frame.panels[0].coherence_stamp.inputs[0].ref
            == next_snapshot.ref
        )

        revised_overview = composer.compose_faceted(
            next_snapshot,
            display=revised_display,
            provenance=provenance,
        )
        assert revised_overview.figure is not overview.figure
        assert revised_overview.figure.evaluated.inputs[0].ref == next_snapshot.ref
        assert composer._faceted_overview_renderer is overview_renderer
        assert overview_renderer._figure is overview_surface
        assert tuple(id(axis) for axis in overview_renderer._axes) == overview_axes
        assert tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in overview_renderer._cells
        ) == overview_artists
        assert (
            revised_overview.overview.raster.pixels
            != overview.overview.raster.pixels
        )
    finally:
        composer.close()


def test_faceted_image_overview_reconciles_exact_revision_in_place() -> None:
    snapshot, view, display = _image_source(1)
    next_snapshot, _view, _display = _image_source(2)
    provenance = PanelProvenance("run", "epoch", "0" * 64)
    composer = PanelComposer(
        "faceted-lifecycle-image",
        intent=ViewIntent.IMAGE,
        size_name="1x2",
        view=view,
    )
    try:
        first = composer.compose_faceted(
            snapshot,
            display=display,
            provenance=provenance,
        )
        renderer = composer._faceted_overview_renderer
        assert renderer is not None and renderer._figure is not None
        figure = renderer._figure
        axes = tuple(id(axis) for axis in renderer._axes)
        artists = tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in renderer._cells
        )
        revised = composer.compose_faceted(
            next_snapshot,
            display=replace(display, revision=1),
            provenance=provenance,
        )
        assert renderer._figure is figure
        assert tuple(id(axis) for axis in renderer._axes) == axes
        assert tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in renderer._cells
        ) == artists
        assert revised.figure.evaluated.inputs[0].ref == next_snapshot.ref
        assert revised.overview.raster.pixels != first.overview.raster.pixels
    finally:
        composer.close()


def test_faceted_meter_overview_reconciles_exact_revision_in_place() -> None:
    snapshot, view, display = _meter_source(1)
    next_snapshot, _view, _display = _meter_source(2)
    provenance = PanelProvenance("run", "epoch", "0" * 64)
    composer = PanelComposer(
        "faceted-lifecycle-meter",
        intent=ViewIntent.METER,
        size_name="1x2",
        view=view,
    )
    try:
        first = composer.compose_faceted(
            snapshot,
            display=display,
            provenance=provenance,
        )
        renderer = composer._faceted_overview_renderer
        assert renderer is not None and renderer._figure is not None
        figure = renderer._figure
        axes = tuple(id(axis) for axis in renderer._axes)
        artists = tuple(
            tuple(id(artist) for artist in cell.source)
            for cell in renderer._cells
        )
        text_before = tuple(cell.source[0].get_text() for cell in renderer._cells)

        revised = composer.compose_faceted(
            next_snapshot,
            display=replace(display, revision=1),
            provenance=provenance,
        )

        assert renderer._figure is figure
        assert tuple(id(axis) for axis in renderer._axes) == axes
        assert tuple(
            tuple(id(artist) for artist in cell.source)
            for cell in renderer._cells
        ) == artists
        assert tuple(
            cell.source[0].get_text() for cell in renderer._cells
        ) != text_before
        assert revised.figure.evaluated.inputs[0].ref == next_snapshot.ref
        assert revised.overview.raster.pixels != first.overview.raster.pixels
    finally:
        composer.close()


def test_faceted_curve_applies_typed_authored_view_without_rebuilding() -> None:
    snapshot, view, _display = _curve_source(1)
    provenance = PanelProvenance("run", "epoch", "0" * 64)
    composer = PanelComposer(
        "faceted-lifecycle-curve-view",
        intent=ViewIntent.CURVE,
        size_name="1x2",
        view=view,
    )
    try:
        composer.compose_faceted(
            snapshot,
            display=CurveDisplayState(),
            provenance=provenance,
        )
        renderer = composer._faceted_overview_renderer
        assert renderer is not None
        axes = tuple(renderer._axes)
        artists = tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in renderer._cells
        )
        composer.compose_faceted(
            snapshot,
            display=CurveDisplayState(
                revision=1,
                relim_mode=RelimMode.FIXED,
                x_view=(1.0, 2.0),
                fixed_y_limits=(-5.0, 5.0),
            ),
            provenance=provenance,
        )
        assert tuple(renderer._axes) == axes
        assert tuple(tuple(axis.get_xlim()) for axis in axes) == ((1.0, 2.0),) * 2
        assert tuple(tuple(axis.get_ylim()) for axis in axes) == ((-5.0, 5.0),) * 2
        assert tuple(
            tuple(
                id(artist)
                for artist in cell.dynamic_artists()
                if artist is not None
            )
            for cell in renderer._cells
        ) == artists
    finally:
        composer.close()

    rejecting = PanelComposer(
        "faceted-lifecycle-curve-type",
        intent=ViewIntent.CURVE,
        size_name="1x2",
        view=view,
    )
    try:
        with pytest.raises(TypeError, match="CurveDisplayState"):
            rejecting.compose_faceted(
                snapshot,
                display=ImageDisplayState(),
                provenance=provenance,
            )
    finally:
        rejecting.close()


def test_faceted_histogram_and_image_use_shared_limit_continuity() -> None:
    provenance = PanelProvenance("run", "epoch", "0" * 64)
    histogram_snapshot, histogram_view, histogram_display = _histogram_source(1)
    histogram = PanelComposer(
        "faceted-lifecycle-histogram-limits",
        intent=ViewIntent.HISTOGRAM,
        size_name="1x2",
        view=histogram_view,
    )
    try:
        histogram.compose_faceted(
            histogram_snapshot,
            display=histogram_display,
            provenance=provenance,
        )
        renderer = histogram._faceted_overview_renderer
        assert renderer is not None
        axes = tuple(renderer._axes)
        artists = tuple(
            tuple(id(artist) for artist in cell.source)
            for cell in renderer._cells
        )
        fixed = FacetedHistogramDisplayState(
            HistogramDisplayState(
                revision=1,
                relim_mode=RelimMode.FIXED,
                fixed_count_limits=(0.0, 20.0),
            )
        )
        histogram.compose_faceted(
            histogram_snapshot,
            display=fixed,
            provenance=provenance,
        )
        assert renderer._histogram_count_limits == (0.0, 20.0)
        assert tuple(tuple(axis.get_ylim()) for axis in axes) == ((0.0, 20.0),) * 2
        logarithmic = FacetedHistogramDisplayState(
            HistogramDisplayState(
                revision=2,
                relim_mode=RelimMode.TIGHT,
                count_scale=HistogramCountScale.LOG,
            )
        )
        histogram.compose_faceted(
            histogram_snapshot,
            display=logarithmic,
            provenance=provenance,
        )
        assert renderer._histogram_count_scale is HistogramCountScale.LOG
        assert renderer._histogram_count_limits[0] == 0.5
        assert tuple(renderer._axes) == axes
        assert tuple(
            tuple(id(artist) for artist in cell.source)
            for cell in renderer._cells
        ) == artists
    finally:
        histogram.close()

    image_snapshot, image_view, image_display = _image_source(1)
    next_image_snapshot, _view, _display = _image_source(2)
    image = PanelComposer(
        "faceted-lifecycle-image-limits",
        intent=ViewIntent.IMAGE,
        size_name="1x2",
        view=image_view,
    )
    try:
        image.compose_faceted(
            image_snapshot,
            display=image_display,
            provenance=provenance,
        )
        renderer = image._faceted_overview_renderer
        assert renderer is not None
        first_limits = renderer._image_color_limits
        axes = tuple(renderer._axes)
        artists = tuple(id(cell.source[0]) for cell in renderer._cells)
        image.compose_faceted(
            next_image_snapshot,
            display=replace(image_display, revision=1),
            provenance=provenance,
        )
        assert renderer._image_color_limits != first_limits
        fixed = ImageDisplayState(
            revision=2,
            relim_mode=RelimMode.FIXED,
            fixed_color_limits=(0.0, 500.0),
        )
        image.compose_faceted(
            next_image_snapshot,
            display=fixed,
            provenance=provenance,
        )
        assert renderer._image_color_limits == (0.0, 500.0)
        assert tuple(renderer._axes) == axes
        assert tuple(id(cell.source[0]) for cell in renderer._cells) == artists
    finally:
        image.close()
