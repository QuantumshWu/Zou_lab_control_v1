"""Focused pure/display-worker contracts for interactive CURVE fronts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    MONITOR_HISTORY,
    StreamGenerationId,
)
from zlc_frontend.curve_display import (
    CurveDisplayState,
    CurveViewportTransform,
    curve_display_form_values,
    curve_display_from_form,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from zlc_frontend.display_range import (
    RelimMode,
    deadband_display_range,
    target_display_range,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedAxis,
    EvaluatedCell,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedInput,
    EvaluatedLayer,
    EvaluatedSeries,
    FigureDocument,
    FigureLayer,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.matplotlib_render import (
    SinglePanelAggRenderer,
)
from zlc_frontend.render import CurvePanelPayload


_SCHEMA = "a" * 64


def _axis(coordinates=(0.0, 1.0, 2.0)) -> EvaluatedAxis:
    return EvaluatedAxis(
        AxisId("history"),
        "Shots ago",
        MONITOR_HISTORY,
        "shot",
        tuple(range(len(coordinates))),
        tuple(coordinates),
    )


def _document_and_data(
    *curves: EvaluatedCurve,
) -> tuple[FigureDocument, EvaluatedFigureData]:
    dataset_id = DatasetId("monitor")
    view = ViewSpec(
        _SCHEMA,
        ViewIntent.CURVE,
        (AxisViewBinding(curves[0].x_axis.axis_id, AxisViewRole.X),),
    )
    document = FigureDocument(
        "curve-document",
        2,
        (DatasetDescriptor(dataset_id, "ROI count", _SCHEMA),),
        (FigureLayer("roi-count", dataset_id, view),),
    )
    evaluated_input = EvaluatedInput(
        dataset_id,
        DatasetRevisionRef(
            BlockId("monitor-block"),
            StreamGenerationId("monitor-generation"),
            _SCHEMA,
            DatasetRevision(7),
        ),
    )
    evaluated = EvaluatedFigureData(
        document.document_id,
        document.revision,
        (evaluated_input,),
        (
            EvaluatedLayer(
                "roi-count",
                dataset_id,
                (
                    EvaluatedCell(
                        (),
                        tuple(EvaluatedSeries((), curve) for curve in curves),
                    ),
                ),
            ),
        ),
    )
    return document, evaluated


def test_shared_relim_targets_and_deadbands_are_exact() -> None:
    assert target_display_range(RelimMode.TIGHT, 10.0, 20.0) == (9.0, 21.0)
    assert target_display_range(RelimMode.TIGHT, 5.0, 5.0) == (4.5, 5.5)
    assert target_display_range(RelimMode.TIGHT, 0.0, 0.0) == (-0.1, 0.1)
    assert target_display_range(RelimMode.NORMAL, 10.0, 20.0) == (0.0, 24.0)
    assert target_display_range(RelimMode.NORMAL, -5.0, 5.0) == (-6.0, 6.0)
    assert deadband_display_range(
        RelimMode.NORMAL,
        (0.0, 120.0),
        0.0,
        100.0,
    ) == (0.0, 120.0)
    assert deadband_display_range(
        RelimMode.TIGHT,
        (-1.0, 11.0),
        0.1,
        9.9,
    ) == (-1.0, 11.0)
    assert deadband_display_range(
        RelimMode.FIXED,
        None,
        -100.0,
        100.0,
        fixed_range=(2.0, 4.0),
    ) == (2.0, 4.0)


def test_curve_form_freezes_painted_fixed_limits_and_noop_keeps_revision() -> None:
    base = CurveDisplayState()
    values = curve_display_form_values(base)
    assert curve_display_from_form(base, values) is base

    values["relim_mode"] = RelimMode.FIXED
    values["y_min"] = -999.0
    values["y_max"] = 999.0
    fixed = curve_display_from_form(
        base,
        values,
        current_y_limits=(-2.0, 8.0),
    )
    assert fixed.revision == 1
    assert fixed.fixed_y_limits == (-2.0, 8.0)
    assert curve_display_from_form(
        fixed,
        curve_display_form_values(fixed),
        current_y_limits=(0.0, 1.0),
    ) is fixed


@pytest.mark.parametrize(
    "coordinates",
    (
        (0.0, 1.0, 2.0),
        (2.0, 1.0, 0.0),
        (0.0, 0.25, 4.0),
        (5.0,),
    ),
)
def test_curve_axis_accepts_monotonic_irregular_and_singleton(coordinates) -> None:
    axis = _axis(coordinates)
    validated = numeric_curve_coordinates(axis)
    assert validated is axis.coordinates
    assert validated == coordinates
    low, high = curve_home_x_limits(axis)
    if len(coordinates) == 1:
        assert low < coordinates[0] < high
    else:
        assert (low, high) == (min(coordinates), max(coordinates))


def test_curve_axis_interaction_fails_closed_for_ambiguous_coordinates() -> None:
    with pytest.raises(TypeError, match="numeric scalar"):
        numeric_curve_coordinates(_axis(("new", "old")))
    with pytest.raises(ValueError, match="strictly monotonic"):
        numeric_curve_coordinates(_axis((0.0, 1.0, 0.5)))
    with pytest.raises(ValueError, match="strictly monotonic"):
        numeric_curve_coordinates(_axis((0.0, 0.0)))


def test_curve_viewport_exact_top_origin_mapping_zoom_pan_and_span() -> None:
    viewport = CurveViewportTransform(
        _axis((0.0, 2.0, 5.0)),
        3,
        (0.1, 0.2, 0.9, 0.8),
        (0.0, 10.0),
        (-2.0, 8.0),
        (-0.25, 5.25),
    )
    assert viewport.widget_normalized_to_data(0.5, 0.5) == pytest.approx(
        (5.0, 3.0)
    )
    assert viewport.data_to_widget_normalized(5.0, 3.0) == pytest.approx(
        (0.5, 0.5)
    )
    assert viewport.zoomed_x_limits(5.0, 0.5) == (2.5, 7.5)
    assert viewport.panned_x_limits(0.3, 0.5) == pytest.approx((-2.5, 7.5))
    assert viewport.selection_x_span(0.7, 0.3) == pytest.approx((2.5, 7.5))


def test_interactive_renderer_returns_exact_bbox_payload_and_shared_axis() -> None:
    axis = _axis((0.0, 1.0, 3.0))
    curve = EvaluatedCurve(
        axis,
        "count",
        np.asarray((0.0, 5.0, 10.0)),
        np.asarray((True, True, True)),
    )
    document, evaluated = _document_and_data(curve)
    renderer = SinglePanelAggRenderer(document, width=320, height=240)
    try:
        raster, payload = renderer.render_interactive_curve(
            evaluated,
            CurveDisplayState(revision=4),
            current_y_limits=None,
            previous_relim_mode=None,
        )
        assert len(raster.pixels) == raster.width * raster.height * 4
        assert isinstance(payload, CurvePanelPayload)
        assert payload.series[0].data is curve
        assert payload.evaluated_input is evaluated.inputs[0]
        assert payload.viewport.display_revision == 4
        assert payload.viewport.x_axis is axis
        assert payload.viewport.y_limits == (0.0, 12.0)
        assert payload.value_unit == "count"
        left, top, right, bottom = payload.viewport.plot_bounds
        assert 0.0 <= left < right <= 1.0
        assert 0.0 <= top < bottom <= 1.0
        x0, y0, width, height = renderer._axis.bbox.bounds
        assert payload.viewport.plot_bounds == pytest.approx(
            (
                x0 / raster.width,
                1.0 - (y0 + height) / raster.height,
                (x0 + width) / raster.width,
                1.0 - y0 / raster.height,
            )
        )
        assert renderer._axis.get_ylabel() == "Signal"
        assert renderer._artists[0].get_color().lower() == "#808080"

        changed_axis = replace(axis, coordinates=(10.0, 20.0, 40.0))
        changed_curve = EvaluatedCurve(
            changed_axis,
            "count",
            curve.values,
            curve.validity,
        )
        _same_document, changed = _document_and_data(changed_curve)
        _raster, changed_payload = renderer.render_interactive_curve(
            changed,
            CurveDisplayState(revision=5),
            current_y_limits=payload.viewport.y_limits,
            previous_relim_mode=RelimMode.TIGHT,
        )
        assert changed_payload.viewport.x_axis.coordinates == (10.0, 20.0, 40.0)
    finally:
        renderer.close()

def test_curve_payload_and_renderer_reject_different_series_axes() -> None:
    first = EvaluatedCurve(
        _axis((0.0, 1.0)),
        None,
        np.asarray((1.0, 2.0)),
        np.asarray((True, True)),
    )
    second = EvaluatedCurve(
        _axis((0.0, 2.0)),
        None,
        np.asarray((3.0, 4.0)),
        np.asarray((True, True)),
    )
    document, evaluated = _document_and_data(first, second)
    renderer = SinglePanelAggRenderer(document, width=240, height=180)
    try:
        with pytest.raises(ValueError, match="share one exact x axis"):
            renderer.render_interactive_curve(
                evaluated,
                CurveDisplayState(),
                current_y_limits=None,
                previous_relim_mode=None,
            )
        _valid_document, valid = _document_and_data(first, replace(first))
        _raster, payload = renderer.render_interactive_curve(
            valid,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
        )
        assert len(payload.series) == 2
    finally:
        renderer.close()
def test_interactive_renderer_keeps_all_same_axis_series_in_payload() -> None:
    axis = _axis((0.0, 1.0, 4.0))
    curves = (
        EvaluatedCurve(
            axis,
            "count",
            np.asarray((1.0, 2.0, 3.0)),
            np.asarray((True, True, True)),
        ),
        EvaluatedCurve(
            axis,
            "count",
            np.asarray((10.0, 20.0, 30.0)),
            np.asarray((True, False, True)),
        ),
    )
    document, evaluated = _document_and_data(*curves)
    renderer = SinglePanelAggRenderer(document, width=240, height=180)
    try:
        _raster, payload = renderer.render_interactive_curve(
            evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
        )
        assert tuple(series.data for series in payload.series) == curves
        assert len(payload.series_labels) == 2
        assert payload.viewport.y_limits == pytest.approx((0.0, 36.0))
    finally:
        renderer.close()
