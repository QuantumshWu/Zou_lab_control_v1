"""Pure W7 saved radial-fit projections shared by Qt and canonical Agg."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitResultBatch,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_spec_for,
)
from zlc_data.fit_model import evaluate_fit_model
from zlc_frontend import DataFigure, FitGridModel, RadialGaussianImageFitPanel
from zlc_frontend.image_display import ImageColormap, ImageDisplayState
from zlc_frontend.figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    suggest_fit_view,
)


def _axis(name, role, size, coordinates=None, *, frame=None, unit=None):
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        unit,
        frame,
    )


def _sparse_radial_figure():
    repeat = _axis("repeat", REPEAT, 2)
    event = _axis("event", SCAN_POINT, 3)
    frame = CoordinateFrameId("camera-sensor")
    y_axis = _axis("camera.y", SPATIAL_Y, 6, frame=frame, unit="pixel")
    x_axis = _axis("camera.x", SPATIAL_X, 8, frame=frame, unit="pixel")
    point_layout = PointLayout.explicit((3,), ((2,), (0,)))
    y_values, x_values = np.meshgrid(
        np.arange(y_axis.size),
        np.arange(x_axis.size),
        indexing="ij",
    )
    image = evaluate_fit_model(
        "radial_gaussian_center",
        (x_values, y_values),
        (10.0, 1.0, 2.0, 3.0, 2.0),
    )
    values = np.stack(
        tuple(image * (1.0 + 0.05 * index) for index in range(4))
    ).reshape(2, 2, y_axis.size, x_axis.size)
    schema = DatasetSchema(
        repeat,
        (event,),
        point_layout,
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.value(),
            np.dtype("<f8"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId("w7-radial-projection"),
        DatasetRevision(1),
        values,
        CellValidity(np.asarray(((True, False), (True, True)), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("w7-radial-projection-generation")),
        block,
    )
    result = bind_fit(
        fit_spec_for(schema, "radial_gaussian_center"),
        schema,
    ).run(snapshot)
    model = FitGridModel.from_result("fit-result/" + "a" * 64, result)
    page = model.page()
    suggestion = suggest_fit_view(
        schema,
        result,
        page.selection,
        page.preferences,
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "w7-radial-projection",
        0,
        (DatasetDescriptor(dataset_id, "saved fit", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    figure = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results={"data": result},
        evaluation_memory_limit_bytes=200 << 20,
        render_memory_limit_bytes=200 << 20,
    )
    return figure, model, result


def test_radial_fit_panels_preserve_sparse_cells_axes_geometry_and_focus_summary():
    figure, model, result = _sparse_radial_figure()
    panels = figure.radial_gaussian_image_fit_panels(
        "data",
        artifact_identity=model.artifact_identity,
    )

    assert len(panels) == 6
    assert all(isinstance(panel, RadialGaussianImageFitPanel) for panel in panels)
    assert {panel.fit_storage_index for panel in panels} == {None, 0, 1, 2, 3}
    assert sum(panel.fit_storage_index is None for panel in panels) == 2
    assert all(panel.evaluated_input.ref == result.source_ref for panel in panels)
    assert all(panel.home_viewport.viewport_revision == 0 for panel in panels)
    assert all(panel.home_viewport is panels[0].home_viewport for panel in panels)
    assert all(
        panel.home_viewport.raster_shape == panel.image.values.shape
        for panel in panels
    )

    for panel in panels:
        overlay = panel.fit_overlay
        assert overlay.artifact_identity == model.artifact_identity
        assert overlay.caption == panel.caption
        assert model.storage_index_or_none(panel.selection) == panel.fit_storage_index
        if panel.fit_storage_index is None:
            assert overlay.status is None
            assert overlay.center_xy is None
            assert overlay.one_over_e_radius is None
            assert overlay.diagnostic == "NOT_PRESENT"
            assert "NOT_PRESENT" in panel.summary
            assert "no neighbouring row was substituted" in panel.summary
        elif overlay.status is FitBatchStatus.CONVERGED:
            summary = model.cell_summary(result, panel.selection)
            assert panel.summary == summary.text
            assert overlay.status is result.statuses[panel.fit_storage_index]
            assert overlay.center_xy is not None
            assert overlay.one_over_e_radius is not None
        else:
            summary = model.cell_summary(result, panel.selection)
            assert panel.summary == summary.text
            assert overlay.status is result.statuses[panel.fit_storage_index]
            assert overlay.center_xy is None
            assert overlay.one_over_e_radius is None
            assert overlay.diagnostic.startswith(overlay.status.value)


def test_pooled_image_range_ignores_invalid_and_nonfinite_components():
    from zlc_frontend.figure import EvaluatedImage
    from zlc_frontend.image_raster import evaluated_image_data_range

    figure, model, _result = _sparse_radial_figure()
    template = figure.radial_gaussian_image_fit_panels(
        "data",
        artifact_identity=model.artifact_identity,
    )[0].image
    first = np.full(template.values.shape, np.nan)
    first[0, 0] = -3.0
    first[0, 1] = 10_000.0
    first_validity = np.zeros(template.values.shape, dtype=bool)
    first_validity[0, :2] = True
    first_validity[0, 1] = False
    second = np.full(template.values.shape, np.inf)
    second[1, 1] = 7.0
    second_validity = np.ones(template.values.shape, dtype=bool)
    images = (
        EvaluatedImage(template.x_axis, template.y_axis, first, first_validity),
        EvaluatedImage(template.x_axis, template.y_axis, second, second_validity),
    )

    assert evaluated_image_data_range(images) == (-3.0, 7.0)
    assert evaluated_image_data_range(
        (
            EvaluatedImage(
                template.x_axis,
                template.y_axis,
                np.full(template.values.shape, np.nan),
                np.ones(template.values.shape, dtype=bool),
            ),
        )
    ) is None


def test_canonical_agg_uses_same_center_ring_projection_without_model_evaluation(
    monkeypatch,
):
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Circle

    from zlc_frontend.matplotlib_render import release_agg_figure

    figure, _model, result = _sparse_radial_figure()

    def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError("radial saved-fit rendering must not evaluate a predicted image")

    monkeypatch.setattr(FitResultBatch, "evaluate_batch", forbidden_evaluate)
    rendered = figure.render(memory_limit_bytes=200 << 20)
    try:
        data_axes = rendered.axes[:6]
        expected_converged = sum(
            status is FitBatchStatus.CONVERGED for status in result.statuses
        )
        assert sum(
            any(isinstance(patch, Circle) for patch in axis.patches)
            for axis in data_axes
        ) == expected_converged
        assert sum(
            any(isinstance(collection, PathCollection) for collection in axis.collections)
            for axis in data_axes
        ) == expected_converged
        assert sum(
            any(
                "NOT_PRESENT" in text.get_text()
                or "NO_VALID_DATA" in text.get_text()
                for text in axis.texts
            )
            for axis in data_axes
        ) == len(data_axes) - expected_converged
        image_meshes = tuple(axis.collections[0] for axis in data_axes)
        assert {mesh.get_cmap().name for mesh in image_meshes} == {"gray"}
        assert all(mesh.get_rasterized() for mesh in image_meshes)
        assert len({mesh.get_clim() for mesh in image_meshes}) == 1
        assert all(axis.get_aspect() == 1.0 for axis in data_axes)
        assert all(axis.yaxis_inverted() for axis in data_axes)
    finally:
        release_agg_figure(rendered)


def test_projected_radial_export_honors_current_view_cmap_clim_and_formats(
    monkeypatch,
):
    from zlc_frontend.matplotlib_render import (
        estimate_projected_radial_fit_render_peak_nbytes,
        release_agg_figure,
        render_radial_gaussian_image_fit_panels,
        save_radial_gaussian_image_fit_panels,
    )

    figure, model, _result = _sparse_radial_figure()
    panels = figure.radial_gaussian_image_fit_panels(
        "data",
        artifact_identity=model.artifact_identity,
    )
    display = ImageDisplayState(
        revision=1,
        colormap=ImageColormap.MAGMA,
        x_view=(1.5, 5.5),
        y_view=(0.5, 4.5),
    )
    limits = (-2.0, 22.0)

    def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError("typed radial export must not evaluate a fit model")

    monkeypatch.setattr(FitResultBatch, "evaluate_batch", forbidden_evaluate)
    required = estimate_projected_radial_fit_render_peak_nbytes(
        panels,
        dpi=72.0,
        columns=3,
    )
    with pytest.raises(ValueError, match="columns cannot exceed"):
        estimate_projected_radial_fit_render_peak_nbytes(
            panels,
            dpi=72.0,
            columns=len(panels) + 1,
        )
    with pytest.raises(MemoryError, match="projected radial render peak"):
        render_radial_gaussian_image_fit_panels(
            panels,
            display,
            limits,
            columns=3,
            dpi=72.0,
            memory_limit_bytes=required - 1,
        )
    rendered = render_radial_gaussian_image_fit_panels(
        panels,
        display,
        limits,
        columns=3,
        dpi=72.0,
        memory_limit_bytes=required,
    )
    try:
        from matplotlib.patches import Rectangle

        data_axes = rendered.axes[: len(panels)]
        assert all(axis.get_xlim() == (1.5, 5.5) for axis in data_axes)
        assert all(axis.get_ylim() == (4.5, 0.5) for axis in data_axes)
        assert all(axis.get_aspect() == 1.0 for axis in data_axes)
        assert {
            axis.collections[0].get_cmap().name for axis in data_axes
        } == {"magma"}
        assert all(axis.collections[0].get_rasterized() for axis in data_axes)
        assert {
            axis.collections[0].get_clim() for axis in data_axes
        } == {limits}
        assert all(" $\\cdot$ " in axis.get_title() for axis in data_axes)
        # Pointer-drag rectangles are uncommitted interaction drafts and are
        # deliberately not part of an authored ImageDisplayState export.
        assert all(
            not any(isinstance(patch, Rectangle) for patch in axis.patches)
            for axis in data_axes
        )
    finally:
        release_agg_figure(rendered)

    signatures = {
        "png": b"\x89PNG\r\n\x1a\n",
        "pdf": b"%PDF",
        "svg": b"<?xml",
        "jpg": b"\xff\xd8",
    }
    for image_format, signature in signatures.items():
        output = BytesIO()
        save_radial_gaussian_image_fit_panels(
            panels[:1],
            display,
            limits,
            output,
            image_format=image_format,
            columns=1,
            dpi=72.0,
            memory_limit_bytes=required,
        )
        assert output.getvalue().startswith(signature)
