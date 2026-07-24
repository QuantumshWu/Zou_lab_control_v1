"""Headless contracts for exact single-panel fit replay presentation."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
import threading

import numpy as np
import pytest

from zlc_data import (
    COMPONENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    bind_fit,
    commit_transform,
    fit_spec_for,
    Selection,
)
from zlc_frontend import (
    CurveDisplayState,
    CurveFitOverlay,
    CurvePanelPayload,
    DataFigure,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
    suggest_fit_view,
    suggest_view,
)
from zlc_frontend.matplotlib_render import SinglePanelAggRenderer
from zlc_frontend.fit_curve_projection import (
    materialize_curve_fit_overlay_plan,
)


def _curve_fixture(
    *,
    invalidate_second: bool = False,
    committed_range: tuple[int, int] | None = None,
):
    repeat = AxisSpec(AxisId("u03d.repeat"), "Repeat", REPEAT, 1, (0,))
    x_coordinates = tuple(float(value) for value in np.linspace(-3.0, 3.0, 31))
    scan = AxisSpec(
        AxisId("u03d.detuning"),
        "Detuning",
        SCAN_POINT,
        len(x_coordinates),
        x_coordinates,
        "MHz",
    )
    component = AxisSpec(
        AxisId("u03d.component"),
        "Component",
        COMPONENT,
        2,
        ("signal", "reference"),
    )
    x = np.asarray(x_coordinates)
    values = np.stack(
        (
            1.0 + 4.0 * np.exp(-((x + 0.4) ** 2) / (2.0 * 0.7**2)),
            3.0 + 1.5 * np.exp(-((x - 0.8) ** 2) / (2.0 * 1.1**2)),
        ),
        axis=-1,
    )[None, :, :]
    validity = np.ones(values.shape, dtype=np.bool_)
    if invalidate_second:
        validity[..., 1] = False
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (component,),
            ValidityContract.components(component.axis_id),
            values.dtype,
            value_unit="count",
        ),
    )
    block = DataBlock(
        BlockId("u03d-curve"),
        DatasetRevision(7),
        values,
        ComponentValidity((component.axis_id,), validity),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03d-curve-generation")),
        block,
    )
    committed = (
        None
        if committed_range is None
        else commit_transform(
            schema,
            DataTransformSpec(
                (Selection.index_range(scan.axis_id, *committed_range),)
            ),
        )
    )
    result = bind_fit(
        fit_spec_for(
            schema,
            "gaussian_offset",
            committed_transform=committed,
        ),
        schema,
    ).run(snapshot)
    dataset_id = DatasetId("u03d-curve-source")
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(repeat.axis_id, AxisViewRole.BATCH),
            AxisViewBinding(scan.axis_id, AxisViewRole.X),
            AxisViewBinding(component.axis_id, AxisViewRole.BATCH),
        ),
    )
    document = FigureDocument(
        "u03d-curve-document",
        4,
        (DatasetDescriptor(dataset_id, "curve", schema.fingerprint),),
        (FigureLayer("curve", dataset_id, view),),
    )
    base = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )
    return base, result


def _radial_fixture(*, committed_box: bool = False):
    repeat = AxisSpec(AxisId("u03d.image.repeat"), "Repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId("u03d.image.point"), "Point", SCAN_POINT, 1, (0,))
    frame = CoordinateFrameId("u03d.camera")
    y_axis = AxisSpec(
        AxisId("u03d.image.y"),
        "Y",
        SPATIAL_Y,
        9,
        tuple(float(value) for value in range(9)),
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("u03d.image.x"),
        "X",
        SPATIAL_X,
        11,
        tuple(float(value) for value in range(11)),
        "pixel",
        frame,
    )
    yy, xx = np.meshgrid(
        np.arange(y_axis.size, dtype=np.float64),
        np.arange(x_axis.size, dtype=np.float64),
        indexing="ij",
    )
    image = 2.0 + 12.0 * np.exp(-((xx - 5.0) ** 2 + (yy - 4.0) ** 2) / 8.0)
    values = image[None, None, :, :]
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            values.dtype,
            value_unit="count",
        ),
    )
    validity = np.ones(values.shape, dtype=np.bool_)
    block = DataBlock(
        BlockId("u03d-image"),
        DatasetRevision(2),
        values,
        ComponentValidity((y_axis.axis_id, x_axis.axis_id), validity),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03d-image-generation")),
        block,
    )
    committed = None
    if committed_box:
        committed = commit_transform(
            schema,
            DataTransformSpec(
                (
                    Selection.rectangle(
                        x_axis.axis_id,
                        y_axis.axis_id,
                        2.0,
                        8.0,
                        2.0,
                        6.0,
                        coordinate_frame=frame,
                    ),
                )
            ),
        )
    result = bind_fit(
        fit_spec_for(
            schema,
            "radial_gaussian_center",
            committed_transform=committed,
        ),
        schema,
    ).run(snapshot)
    suggestion = (
        suggest_view(schema, ViewIntent.IMAGE)
        if committed_box
        else suggest_fit_view(schema, result)
    )
    assert suggestion.spec is not None
    dataset_id = DatasetId("u03d-image-source")
    document = FigureDocument(
        "u03d-image-document",
        1,
        (DatasetDescriptor(dataset_id, "image", schema.fingerprint),),
        (FigureLayer("image", dataset_id, suggestion.spec),),
    )
    figure = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results=None if committed_box else {"image": result},
    )
    return figure, result


def test_clone_reuses_exact_evaluation_and_curve_projection_maps_every_batch():
    base, result = _curve_fixture()
    fitted = base.with_fit_results({"curve": result})
    cleared = fitted.with_fit_results(None)

    assert fitted is not base
    assert fitted.evaluated is base.evaluated is cleared.evaluated
    assert (
        fitted.evaluated.layers[0].cells[0].series[0].data.values
        is base.evaluated.layers[0].cells[0].series[0].data.values
    )
    assert not base.has_fit_overlays
    assert fitted.has_fit_overlays
    assert not cleared.has_fit_overlays

    overlays = materialize_curve_fit_overlay_plan(
        fitted.single_panel_curve_fit_overlay_plan(
            result_identity="draft-generation:12",
        )
    )
    series = fitted.evaluated.layers[0].cells[0].series
    assert len(overlays) == len(series) == 2
    assert tuple(item.batch_storage_index for item in overlays) == (0, 1)
    assert all(item.source_ref == result.source_ref for item in overlays)
    assert all(item.result_identity == "draft-generation:12" for item in overlays)
    assert all(item.status is FitBatchStatus.CONVERGED for item in overlays)
    assert all(item.source_sample_span == (0, 31) for item in overlays)
    assert tuple(item.series_batch_address for item in overlays) == tuple(
        item.batch_address for item in series
    )
    for index, (overlay, item) in enumerate(zip(overlays, series, strict=True)):
        expected = result.evaluate_batch(
            index,
            (np.asarray(item.data.x_axis.coordinates, dtype=np.float64),),
        )
        np.testing.assert_allclose(overlay.predicted_y, expected)
        assert not overlay.predicted_y.flags.writeable
    assert not np.array_equal(overlays[0].predicted_y, overlays[1].predicted_y)

    foreign = replace(
        result,
        source_ref=replace(result.source_ref, block_id=BlockId("foreign-curve")),
    )
    with pytest.raises(ValueError, match="source reference differs"):
        base.with_fit_results({"curve": foreign})


def test_projection_preserves_failed_and_not_present_rows_without_row_zero(
    monkeypatch,
):
    base, failed_result = _curve_fixture(invalidate_second=True)
    fitted = base.with_fit_results({"curve": failed_result})
    overlays = materialize_curve_fit_overlay_plan(
        fitted.single_panel_curve_fit_overlay_plan(result_identity="failed:1")
    )
    assert overlays[0].status is FitBatchStatus.CONVERGED
    assert overlays[0].predicted_y.size == 31
    assert overlays[1].status is FitBatchStatus.NO_VALID_DATA
    assert overlays[1].predicted_y.size == 0
    assert overlays[1].diagnostic.startswith("NO_VALID_DATA")

    import zlc_frontend.fit_curve_projection as projection

    original = projection.fit_batch_storage_index
    evaluated_series = fitted.evaluated.layers[0].cells[0].series

    def sparse_mapping(result, layer, cell, series):
        if series is evaluated_series[1]:
            return None
        return original(result, layer, cell, series)

    calls = []
    original_evaluate = type(failed_result).evaluate_batch

    def traced(self, storage_index, coordinates):
        calls.append(storage_index)
        return original_evaluate(self, storage_index, coordinates)

    monkeypatch.setattr(projection, "fit_batch_storage_index", sparse_mapping)
    monkeypatch.setattr(type(failed_result), "evaluate_batch", traced)
    sparse = materialize_curve_fit_overlay_plan(
        fitted.single_panel_curve_fit_overlay_plan(result_identity="sparse:2")
    )
    assert calls == [0]
    assert sparse[1].batch_storage_index is None
    assert sparse[1].status is None
    assert sparse[1].diagnostic == "NOT_PRESENT"
    assert sparse[1].predicted_y.size == 0


def test_interactive_renderer_validates_join_then_failure_clears_old_success():
    base, result = _curve_fixture()
    fitted = base.with_fit_results({"curve": result})
    overlays = materialize_curve_fit_overlay_plan(
        fitted.single_panel_curve_fit_overlay_plan(result_identity="draft:3")
    )
    renderer = SinglePanelAggRenderer(
        fitted.document,
        width=640,
        height=420,
    )
    try:
        _raster, payload = renderer.render_interactive_curve(
            fitted.evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
            fit_overlays=overlays,
        )
        assert isinstance(payload, CurvePanelPayload)
        assert payload.fit_overlays == overlays
        assert all(item.get_visible() for item in renderer._fit_artists)
        previous = tuple(np.asarray(item.get_ydata()).copy() for item in renderer._fit_artists)

        with pytest.raises(ValueError, match="another series address"):
            renderer.render_interactive_curve(
                fitted.evaluated,
                CurveDisplayState(),
                current_y_limits=None,
                previous_relim_mode=None,
                fit_overlays=tuple(reversed(overlays)),
            )
        for artist, expected in zip(renderer._fit_artists, previous, strict=True):
            np.testing.assert_array_equal(artist.get_ydata(), expected)

        empty = np.empty((0,), dtype=np.dtype("<f8"))
        replacement = (
            CurveFitOverlay(
                overlays[0].source_ref,
                "draft:4",
                overlays[0].series_batch_address,
                overlays[0].source_sample_span,
                overlays[0].batch_storage_index,
                FitBatchStatus.SOLVER_FAILED,
                "SOLVER_FAILED: deliberate witness",
                empty,
            ),
            CurveFitOverlay(
                overlays[1].source_ref,
                "draft:4",
                overlays[1].series_batch_address,
                overlays[1].source_sample_span,
                None,
                None,
                "NOT_PRESENT",
                empty,
            ),
        )
        _raster, failed_payload = renderer.render_interactive_curve(
            fitted.evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
            fit_overlays=replacement,
        )
        assert failed_payload.fit_overlays == replacement
        assert all(not item.get_visible() for item in renderer._fit_artists)
        assert all(np.asarray(item.get_ydata()).size == 0 for item in renderer._fit_artists)
        assert "SOLVER_FAILED" in renderer._fit_diagnostic_artists[0].get_text()
        assert "NOT_PRESENT" in renderer._fit_diagnostic_artists[1].get_text()
    finally:
        renderer.close()


def test_radial_single_panel_seam_reuses_named_projection_only():
    figure, result = _radial_fixture()
    overlay = figure.single_panel_radial_fit_overlay(
        result_identity="draft-radial:9",
    )
    assert overlay.source_ref == result.source_ref
    assert overlay.result_identity == "draft-radial:9"
    assert overlay.status is FitBatchStatus.CONVERGED
    assert overlay.center_xy is not None
    assert overlay.one_over_e_radius is not None


def test_transient_selector_fit_stays_on_cached_full_view_and_only_draws_roi():
    base, result = _curve_fixture(committed_range=(6, 25))
    full_curve = base.evaluated.layers[0].cells[0].series[0].data
    with pytest.raises(ValueError, match="committed transform"):
        base.with_fit_results({"curve": result})

    overlays = materialize_curve_fit_overlay_plan(
        base.transient_single_panel_curve_fit_overlay_plan(
            result,
            result_identity="selector-draft:5",
        )
    )
    assert all(item.source_sample_span == (6, 25) for item in overlays)
    assert all(item.predicted_y.shape == (19,) for item in overlays)
    assert full_curve.values.shape == (31,)
    assert base.evaluated.layers[0].cells[0].series[0].data is full_curve

    renderer = SinglePanelAggRenderer(base.document, width=640, height=420)
    try:
        renderer.render_interactive_curve(
            base.evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
            fit_overlays=overlays,
        )
        for artist, series in zip(
            renderer._fit_artists,
            base.evaluated.layers[0].cells[0].series,
            strict=True,
        ):
            np.testing.assert_array_equal(
                artist.get_xdata(),
                np.asarray(series.data.x_axis.coordinates[6:25]),
            )
            assert np.asarray(artist.get_ydata()).shape == (19,)
    finally:
        renderer.close()

    image_base, image_result = _radial_fixture(committed_box=True)
    full_image = image_base.evaluated.layers[0].cells[0].series[0].data
    assert full_image.values.shape == (9, 11)
    with pytest.raises(ValueError, match="committed transform"):
        image_base.with_fit_results({"image": image_result})
    radial = image_base.transient_single_panel_radial_fit_overlay(
        image_result,
        result_identity="selector-radial:6",
    )
    assert radial.status is FitBatchStatus.CONVERGED
    assert radial.result_identity == "selector-radial:6"
    assert image_base.evaluated.layers[0].cells[0].series[0].data is full_image


def test_curve_overlay_materialization_checks_cancel_between_batches(monkeypatch):
    import zlc_workbench.data_figure.render_lane as figure_module

    base, result = _curve_fixture(committed_range=(6, 25))
    state = CurveDisplayState()
    cancelled = threading.Event()
    calls = []
    original = type(result).evaluate_batch

    def traced(self, storage_index, coordinates):
        calls.append(storage_index)
        prediction = original(self, storage_index, coordinates)
        cancelled.set()
        return prediction

    monkeypatch.setattr(type(result), "evaluate_batch", traced)
    with pytest.raises(CancelledError):
        figure_module._render_typed_front(
            base,
            state,
            current_value_limits=None,
            previous_relim_mode=None,
            previous_count_scale=None,
            sequence=1,
            cancelled=cancelled,
            fit_result=result,
            fit_result_identity="cancel:1",
        )
    assert calls == [0]
