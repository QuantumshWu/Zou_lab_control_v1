"""U0.3c exact current-state IMAGE PNG export owner."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DatasetRevision,
    DatasetRevisionRef,
    FitBatchStatus,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import DatasetId, EvaluatedAxis, EvaluatedImage, EvaluatedInput
from zlc_frontend.image_display import (
    ImageColormap,
    ImageDisplayState,
    image_display_for_viewport,
)
from zlc_frontend.image_raster import rasterize_image_indexed8
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.matplotlib_render import (
    estimate_image_png_export_peak_nbytes,
    save_image_panel_png,
)
from zlc_frontend.render import ImagePanelPayload, RadialGaussianImageFitOverlay
from zlc_frontend.render_style import indexed_colormap


def _current_image_payload(*, fit_overlay: bool = False):
    frame = CoordinateFrameId("u03c-camera-frame")
    y_axis = AxisSpec(
        AxisId("camera.y"),
        "camera y",
        SPATIAL_Y,
        3,
        (20.0, 21.0, 22.0),
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("camera.x"),
        "camera x",
        SPATIAL_X,
        4,
        (10.0, 11.0, 12.0, 13.0),
        "pixel",
        frame,
    )
    home = ImageViewportTransform((y_axis, x_axis))
    viewport = home.with_visible_bounds(
        (0.25, 0.0, 0.75, 2.0 / 3.0),
        viewport_revision=4,
    )
    authored_view = image_display_for_viewport(ImageDisplayState(), viewport)
    display = ImageDisplayState(
        revision=viewport.viewport_revision,
        relim_mode=RelimMode.FIXED,
        colormap=ImageColormap.MAGMA,
        x_view=authored_view.x_view,
        y_view=authored_view.y_view,
        fixed_color_limits=(2.0, 10.0),
    )
    values = np.asarray(
        (
            (1.0, 2.0, 3.0, 4.0),
            (5.0, 6.0, 7.0, 8.0),
            (9.0, 10.0, 11.0, 12.0),
        ),
        dtype=np.float64,
    )
    validity = np.ones(values.shape, dtype=np.bool_)
    validity[2, 3] = False
    image = EvaluatedImage(
        x_axis=EvaluatedAxis(
            x_axis.axis_id,
            x_axis.name,
            x_axis.role,
            x_axis.unit,
            tuple(range(x_axis.size)),
            x_axis.coordinates,
            coordinate_frame=frame,
        ),
        y_axis=EvaluatedAxis(
            y_axis.axis_id,
            y_axis.name,
            y_axis.role,
            y_axis.unit,
            tuple(range(y_axis.size)),
            y_axis.coordinates,
            coordinate_frame=frame,
        ),
        values=values,
        validity=validity,
        value_unit="photon",
    )
    ref = DatasetRevisionRef(
        BlockId("u03c-image-block"),
        StreamGenerationId("u03c-image-generation"),
        "a" * 64,
        DatasetRevision(7),
    )
    evaluated_input = EvaluatedInput(DatasetId("camera"), ref)
    raster, data_range, histogram, color_limits = rasterize_image_indexed8(
        image,
        display,
        current_color_limits=None,
        previous_relim_mode=None,
    )
    assert raster.width == x_axis.size and raster.height == y_axis.size
    assert color_limits == display.fixed_color_limits
    overlay = (
        RadialGaussianImageFitOverlay(
            ref,
            "capture-fit/" + "b" * 64,
            0,
            FitBatchStatus.CONVERGED,
            frame,
            "site=(2, 1)",
            "",
            (11.5, 21.0),
            0.6,
        )
        if fit_overlay
        else None
    )
    payload = ImagePanelPayload(
        image=image,
        evaluated_input=evaluated_input,
        viewport=viewport,
        data_range=data_range,
        histogram_counts=histogram,
        base_palette=indexed_colormap(display.colormap.value),
        color_limits=color_limits,
        fit_overlay=overlay,
    )
    assert payload.value_unit == "photon"
    return payload, display


def test_image_panel_png_export_preserves_exact_current_front(monkeypatch):
    import zlc_frontend.matplotlib_render as owner
    from matplotlib.figure import Figure

    payload, display = _current_image_payload()
    required = estimate_image_png_export_peak_nbytes(payload.image, dpi=72.0)
    assert required > payload.image.values.nbytes + payload.image.validity.nbytes

    draw_calls = []
    original_draw = owner._draw_projected_image

    def recording_draw(axis, figure, data, **kwargs):
        draw_calls.append((data, dict(kwargs)))
        original_draw(axis, figure, data, **kwargs)
        assert figure.axes[-1].get_ylabel() == "Value [photon]"

    saved = []
    original_savefig = Figure.savefig

    def recording_savefig(figure, *args, **kwargs):
        data_axis = figure.axes[0]
        saved.append(
            (
                data_axis.get_title(),
                tuple(text.get_text() for text in data_axis.texts),
            )
        )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(owner, "_draw_projected_image", recording_draw)
    monkeypatch.setattr(Figure, "savefig", recording_savefig)

    with pytest.raises(MemoryError, match="image panel PNG export peak"):
        save_image_panel_png(
            payload,
            display,
            BytesIO(),
            dpi=72.0,
            memory_limit_bytes=required - 1,
        )
    assert draw_calls == []

    output = BytesIO()
    save_image_panel_png(
        payload,
        display,
        output,
        dpi=72.0,
        memory_limit_bytes=required,
    )
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(draw_calls) == 1
    exact_image, arguments = draw_calls[0]
    assert exact_image is payload.image
    assert arguments == {
        "colormap": display.colormap.value,
        "color_limits": payload.color_limits,
        "visible_bounds": payload.viewport.visible_bounds,
        "regular_pixel_contract": True,
        "center": None,
        "radius": None,
        "diagnostic": None,
    }
    assert saved == [("", ())]


@pytest.mark.parametrize("descending_x", (False, True))
@pytest.mark.parametrize("descending_y", (False, True))
def test_regular_pixel_export_keeps_first_declared_column_left_and_row_top(
    descending_x: bool,
    descending_y: bool,
) -> None:
    import zlc_frontend.matplotlib_render as owner
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    frame = CoordinateFrameId("orientation-camera")
    x_coordinates = (10.0, 12.0, 14.0)
    y_coordinates = (20.0, 23.0)
    if descending_x:
        x_coordinates = tuple(reversed(x_coordinates))
    if descending_y:
        y_coordinates = tuple(reversed(y_coordinates))
    values = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    image = EvaluatedImage(
        EvaluatedAxis(
            AxisId("orientation.x"),
            "x",
            SPATIAL_X,
            "pixel",
            (0, 1, 2),
            x_coordinates,
            frame,
        ),
        EvaluatedAxis(
            AxisId("orientation.y"),
            "y",
            SPATIAL_Y,
            "pixel",
            (0, 1),
            y_coordinates,
            frame,
        ),
        values,
        np.ones(values.shape, dtype=np.bool_),
    )
    figure = Figure(figsize=(4.0, 3.0), dpi=72.0)
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        owner._draw_projected_image(
            axis,
            figure,
            image,
            colormap="gray",
            color_limits=(1.0, 6.0),
            regular_pixel_contract=True,
            center=None,
            radius=None,
            diagnostic=None,
        )
        canvas.draw()
        first_x = axis.transData.transform((x_coordinates[0], y_coordinates[0]))[0]
        last_x = axis.transData.transform((x_coordinates[-1], y_coordinates[0]))[0]
        first_y = axis.transData.transform((x_coordinates[0], y_coordinates[0]))[1]
        last_y = axis.transData.transform((x_coordinates[0], y_coordinates[-1]))[1]
        assert first_x < last_x
        assert first_y > last_y
        np.testing.assert_array_equal(
            np.ma.getdata(axis.images[0].get_array()),
            values,
        )
    finally:
        owner.release_agg_figure(figure)


def test_canonical_irregular_radial_projection_keeps_exact_cell_edges() -> None:
    import zlc_frontend.matplotlib_render as owner
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    frame = CoordinateFrameId("irregular-camera")
    values = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    image = EvaluatedImage(
        EvaluatedAxis(
            AxisId("irregular.x"),
            "x",
            SPATIAL_X,
            "pixel",
            (0, 1, 2),
            (10.0, 11.0, 15.0),
            frame,
        ),
        EvaluatedAxis(
            AxisId("irregular.y"),
            "y",
            SPATIAL_Y,
            "pixel",
            (0, 1),
            (20.0, 24.0),
            frame,
        ),
        values,
        np.ones(values.shape, dtype=np.bool_),
    )
    figure = Figure(figsize=(4.0, 3.0), dpi=72.0)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        owner._draw_projected_image(
            axis,
            figure,
            image,
            colormap="gray",
            color_limits=(1.0, 6.0),
            regular_pixel_contract=False,
            center=(11.0, 20.0),
            radius=0.5,
            diagnostic=None,
        )
        assert len(axis.images) == 0
        mesh_coordinates = axis.collections[0].get_coordinates()
        np.testing.assert_allclose(mesh_coordinates[0, :, 0], (9.5, 10.5, 13.0, 17.0))
        np.testing.assert_allclose(mesh_coordinates[:, 0, 1], (18.0, 22.0, 26.0))
    finally:
        owner.release_agg_figure(figure)


def test_generic_image_export_draws_fit_overlay_and_preserves_dynamic_clim(
    monkeypatch,
) -> None:
    import zlc_frontend.matplotlib_render as owner
    from matplotlib.axes import Axes

    raw_payload, fixed_display = _current_image_payload()
    fit_payload, _ = _current_image_payload(fit_overlay=True)
    required = estimate_image_png_export_peak_nbytes(raw_payload.image, dpi=72.0)

    viewport_type = type(fit_payload.viewport)
    point_calls = []
    span_calls = []

    def projected_point(self, coordinate_xy, *, coordinate_frame):
        point_calls.append((self, coordinate_xy, coordinate_frame))
        return 0.25, 0.75

    def projected_span(self, span_xy, *, coordinate_frame):
        span_calls.append((self, span_xy, coordinate_frame))
        return 0.4, 0.4

    monkeypatch.setattr(
        viewport_type,
        "unbounded_visible_point_for_coordinate",
        projected_point,
    )
    monkeypatch.setattr(
        viewport_type,
        "visible_span_for_coordinate_span",
        projected_span,
    )

    def contour_is_not_a_radial_overlay(*_args, **_kwargs):
        raise AssertionError("radial IMAGE export must not synthesize a contour")

    monkeypatch.setattr(Axes, "contour", contour_is_not_a_radial_overlay)
    observed_draws = []
    original_draw = owner._draw_projected_image

    def recording_draw(axis, figure, data, **kwargs):
        observed_draws.append(dict(kwargs))
        return original_draw(axis, figure, data, **kwargs)

    monkeypatch.setattr(owner, "_draw_projected_image", recording_draw)
    output = BytesIO()
    save_image_panel_png(
        fit_payload,
        fixed_display,
        output,
        dpi=72.0,
        memory_limit_bytes=required,
    )
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    overlay = fit_payload.fit_overlay
    assert overlay is not None
    assert point_calls == [
        (fit_payload.viewport, overlay.center_xy, overlay.coordinate_frame)
    ]
    assert span_calls == [
        (
            fit_payload.viewport,
            (2.0 * overlay.one_over_e_radius, 2.0 * overlay.one_over_e_radius),
            overlay.coordinate_frame,
        )
    ]
    assert len(observed_draws) == 1
    fit_draw = observed_draws[0]
    assert fit_draw["colormap"] == fixed_display.colormap.value
    assert fit_draw["color_limits"] == fit_payload.color_limits
    assert fit_draw["visible_bounds"] == fit_payload.viewport.visible_bounds
    assert fit_draw["regular_pixel_contract"] is True
    # Forced viewport coordinates prove the export does not pass the
    # authority DTO's physical values straight to Matplotlib.
    assert fit_draw["center"] == pytest.approx((11.0, 21.0))
    assert fit_draw["radius"] == pytest.approx(0.4)
    assert fit_draw["diagnostic"] is None

    dynamic_display = replace(
        fixed_display,
        relim_mode=RelimMode.NORMAL,
        fixed_color_limits=None,
    )
    _raster, data_range, histogram, dynamic_limits = rasterize_image_indexed8(
        raw_payload.image,
        dynamic_display,
        current_color_limits=(0.0, 13.2),
        previous_relim_mode=RelimMode.NORMAL,
    )
    assert dynamic_limits == (0.0, 13.2)
    assert dynamic_limits != raw_payload.color_limits
    dynamic_payload = replace(
        raw_payload,
        data_range=data_range,
        histogram_counts=histogram,
        color_limits=dynamic_limits,
    )
    observed_draws.clear()
    output = BytesIO()
    save_image_panel_png(
        dynamic_payload,
        dynamic_display,
        output,
        dpi=72.0,
        memory_limit_bytes=required,
    )
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert observed_draws[0]["color_limits"] == dynamic_limits
    assert observed_draws[0]["center"] is None
    assert observed_draws[0]["radius"] is None


@pytest.mark.parametrize(
    ("status", "diagnostic"),
    (
        (None, "NOT_PRESENT"),
        (FitBatchStatus.NO_VALID_DATA, "NO_VALID_DATA: masked ROI"),
    ),
)
def test_image_fit_export_failure_or_sparse_cell_has_no_success_geometry(
    monkeypatch,
    status: FitBatchStatus | None,
    diagnostic: str,
) -> None:
    import zlc_frontend.matplotlib_render as owner
    from matplotlib.axes import Axes

    payload, display = _current_image_payload(fit_overlay=True)
    converged = payload.fit_overlay
    assert converged is not None
    overlay = replace(
        converged,
        batch_storage_index=None if status is None else 0,
        status=status,
        diagnostic=diagnostic,
        center_xy=None,
        one_over_e_radius=None,
    )
    payload = replace(payload, fit_overlay=overlay)
    required = estimate_image_png_export_peak_nbytes(payload.image, dpi=72.0)

    def forbidden_geometry(*_args, **_kwargs):
        raise AssertionError("failed/sparse fit export retained success geometry")

    monkeypatch.setattr(Axes, "scatter", forbidden_geometry)
    monkeypatch.setattr(Axes, "contour", forbidden_geometry)
    observed = []
    original_draw = owner._draw_projected_image

    def recording_draw(axis, figure, data, **kwargs):
        observed.append(dict(kwargs))
        return original_draw(axis, figure, data, **kwargs)

    monkeypatch.setattr(owner, "_draw_projected_image", recording_draw)
    output = BytesIO()
    save_image_panel_png(
        payload,
        display,
        output,
        dpi=72.0,
        memory_limit_bytes=required,
    )
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(observed) == 1
    assert observed[0]["center"] is None
    assert observed[0]["radius"] is None
    assert observed[0]["diagnostic"] == diagnostic


def _profile_witness_image(size: int, dtype) -> EvaluatedImage:
    frame = CoordinateFrameId(f"profile-{size}")
    coordinates = tuple(range(size))
    return EvaluatedImage(
        EvaluatedAxis(
            AxisId(f"profile-{size}.x"),
            "x",
            SPATIAL_X,
            "pixel",
            coordinates,
            coordinates,
            frame,
        ),
        EvaluatedAxis(
            AxisId(f"profile-{size}.y"),
            "y",
            SPATIAL_Y,
            "pixel",
            coordinates,
            coordinates,
            frame,
        ),
        np.zeros((size, size), dtype=dtype),
        np.ones((size, size), dtype=np.bool_),
    )


@pytest.mark.parametrize(
    ("size", "dtype", "measured_incremental_peak"),
    (
        (512, np.uint8, 35_367_651),
        (1024, np.uint8, 75_846_083),
        (1024, np.uint16, 76_947_732),
        (1024, np.float32, 78_752_306),
        (1024, np.float64, 82_942_638),
        (2304, np.uint16, 373_728_906),
    ),
)
def test_image_export_estimate_exceeds_frozen_profile_witness(
    size: int,
    dtype,
    measured_incremental_peak: int,
) -> None:
    """Keep the static imshow formula above the July-2026 Windows Agg profiles."""

    image = _profile_witness_image(size, dtype)
    assert estimate_image_png_export_peak_nbytes(image) >= measured_incremental_peak


def test_image_panel_png_export_rejects_state_that_does_not_own_payload_front():
    payload, display = _current_image_payload()
    required = estimate_image_png_export_peak_nbytes(payload.image, dpi=72.0)

    with pytest.raises(ValueError, match="exact payload viewport"):
        save_image_panel_png(
            payload,
            replace(display, x_view=None),
            BytesIO(),
            dpi=72.0,
            memory_limit_bytes=required,
        )
    forged_x_axis = replace(payload.viewport.x_axis, name="forged camera x")
    forged_payload = replace(
        payload,
        viewport=ImageViewportTransform(
            (payload.viewport.y_axis, forged_x_axis),
            viewport_revision=payload.viewport.viewport_revision,
            visible_bounds=payload.viewport.visible_bounds,
        ),
    )
    with pytest.raises(ValueError, match="exact payload viewport"):
        save_image_panel_png(
            forged_payload,
            display,
            BytesIO(),
            dpi=72.0,
            memory_limit_bytes=required,
        )
    with pytest.raises(ValueError, match="colormap differs"):
        save_image_panel_png(
            payload,
            replace(display, colormap=ImageColormap.VIRIDIS),
            BytesIO(),
            dpi=72.0,
            memory_limit_bytes=required,
        )
    with pytest.raises(ValueError, match="fixed image display limits differ"):
        save_image_panel_png(
            payload,
            replace(display, fixed_color_limits=(3.0, 9.0)),
            BytesIO(),
            dpi=72.0,
            memory_limit_bytes=required,
        )
