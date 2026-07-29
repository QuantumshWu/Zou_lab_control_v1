import ast
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_frontend.site_map import immutable_site_state, site_ring_radius


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_site_map_fact_and_exact_view_owners_are_headless() -> None:
    for name in ("site_map.py", "site_map_render.py"):
        tree = ast.parse((ROOT / "zlc_frontend" / name).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        assert roots.isdisjoint({"PyQt5", "matplotlib"})

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_frontend.site_map\n"
                "import zlc_frontend.site_map_render\n"
                "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
                "for name in sys.modules)\n"
                "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
                "for name in sys.modules)\n"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_site_ring_radius_finds_nearest_pair_across_workspace_blocks():
    centers = np.column_stack(
        (np.arange(260, dtype=float) * 100.0, np.zeros(260, dtype=float))
    )
    centers[128, 0] = centers[127, 0] + 10.0

    assert site_ring_radius(centers) == pytest.approx(3.0)


def test_site_ring_radius_preserves_floor_and_duplicate_center_semantics():
    assert site_ring_radius(np.empty((0, 2))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((2.0, 3.0), (2.0, 3.0)))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((0.0, 0.0), (1.0, 0.0)))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((0.0, 0.0), (20.0, 0.0)))) == pytest.approx(6.0)


def test_site_ring_radius_rejects_non_site_matrix_and_bounds_nonfinite_input():
    with pytest.raises(ValueError, match=r"shape \(sites, 2\)"):
        site_ring_radius(np.zeros((3, 3)))

    assert site_ring_radius(np.asarray(((0.0, 0.0), (np.nan, 1.0)))) == pytest.approx(1.5)


def test_immutable_site_state_owns_exact_dtype_shape_and_validity():
    centers = np.asarray(((1, 2), (3, 4), (5, 6), (7, 8)))
    occupied = np.asarray((False, True, False, False), dtype=bool)
    validity = np.asarray((True, True, False, True), dtype=bool)

    frozen_centers, frozen_occupied, frozen_validity = immutable_site_state(
        centers,
        occupied,
        validity,
        site_count=4,
    )
    assert frozen_centers.dtype == np.dtype("<f8")
    assert frozen_centers.shape == (4, 2)
    np.testing.assert_array_equal(frozen_occupied, occupied)
    np.testing.assert_array_equal(frozen_validity, validity)
    assert not frozen_centers.flags.writeable
    assert not frozen_occupied.flags.writeable
    assert not frozen_validity.flags.writeable


def test_immutable_site_state_rejects_wrong_dtype_or_shape():
    with pytest.raises(TypeError, match="bool dtype"):
        immutable_site_state(
            np.zeros((2, 2)),
            np.asarray((0, 1)),
            np.asarray((True, True)),
            site_count=2,
        )
    with pytest.raises(ValueError, match="expected"):
        immutable_site_state(
            np.zeros((2, 2)),
            np.asarray((False,), dtype=bool),
            np.asarray((True, True), dtype=bool),
            site_count=2,
        )


def _site_map_image(values):
    from zlc_data import (
        AxisId,
        AxisSourceRef,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedImage
    from zlc_frontend.image_view import ImageViewportTransform

    frame = CoordinateFrameId("site-map-test-frame")
    x_spec = AxisSpec(
        AxisId("site-map-test-x"), "x", SPATIAL_X, 2,
        (0.0, 1.0), unit="pixel", coordinate_frame=frame,
    )
    y_spec = AxisSpec(
        AxisId("site-map-test-y"), "y", SPATIAL_Y, 2,
        (0.0, 1.0), unit="pixel", coordinate_frame=frame,
    )
    x_axis = EvaluatedAxis(
        AxisSourceRef.tensor(x_spec.axis_id), "x", SPATIAL_X, "pixel",
        (0, 1), (0.0, 1.0), frame,
    )
    y_axis = EvaluatedAxis(
        AxisSourceRef.tensor(y_spec.axis_id), "y", SPATIAL_Y, "pixel",
        (0, 1), (0.0, 1.0), frame,
    )
    values = np.asarray(values, dtype=np.uint8)
    return (
        EvaluatedImage(
            x_axis,
            y_axis,
            values,
            np.ones(values.shape, dtype=bool),
        ),
        ImageViewportTransform((y_spec, x_spec)),
        frame,
    )


def _site_map_inputs(revision):
    from zlc_data import (
        BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId,
    )
    from zlc_frontend.figure import DatasetId, EvaluatedInput

    def one(name, schema):
        return EvaluatedInput(
            DatasetId(name),
            DatasetRevisionRef(
                BlockId(f"{name}-block"),
                StreamGenerationId(f"{name}-generation"),
                schema,
                DatasetRevision(revision),
            ),
        )

    return one("background", "a" * 64), one("state", "b" * 64)


def _site_map_protocol_view(image_projection, identity, inputs, *, cell_index=0):
    from zlc_data import AxisId, AxisSpec, Selection, SITE

    image, viewport, frame = image_projection
    return SimpleNamespace(
        background=image,
        background_input=inputs[0],
        home_viewport=viewport,
        site_axis=AxisSpec(AxisId("site"), "site", SITE, 1, ("A",)),
        coordinate_frame=frame,
        centers_xy=np.zeros((1, 2)),
        site_radius=1.0,
        site_validity=np.ones(1, dtype=bool),
        run_id="run",
        provenance_epoch_id="epoch",
        coherence_identity="coherence",
        summary="summary",
        site_state_input=inputs[1],
        cell_selection=Selection.index(AxisId("repeat"), cell_index),
        site_geometry_identity="geometry",
        view_identity=identity,
        site_state=None,
        presentation_kind="site-map",
        materialize_area_outputs=lambda source, selection: {},
    )


def test_site_map_composer_persists_named_renderer_and_scans_each_image_once(
    monkeypatch,
):
    import zlc_frontend.matplotlib_render as matplotlib_render
    import zlc_frontend.site_map_render as site_map_render
    from zlc_frontend.display_range import RelimMode
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.plot_layout import panel_surface_geometry

    renderer_instances = []

    class FakeRenderer:
        def __init__(self, **options):
            self.options = options
            self.closed = 0
            renderer_instances.append(self)

        def close(self):
            self.closed += 1

    range_inputs = []

    def data_range(images):
        (image,) = tuple(images)
        range_inputs.append(image)
        return float(np.min(image.values)), float(np.max(image.values))

    compose_calls = []

    def compose(view, display, **options):
        compose_calls.append(options)
        return object(), (2.0, 9.0)

    monkeypatch.setattr(
        matplotlib_render,
        "ImagePanelAggRenderer",
        FakeRenderer,
    )
    monkeypatch.setattr(site_map_render, "evaluated_image_data_range", data_range)
    monkeypatch.setattr(site_map_render, "_compose_site_map_front", compose)

    revision_one_inputs = _site_map_inputs(1)
    first = _site_map_protocol_view(
        _site_map_image(((2, 3), (4, 9))),
        "first",
        revision_one_inputs,
    )
    second_frozen_cell = _site_map_protocol_view(
        _site_map_image(((11, 12), (13, 14))),
        "second-frozen-cell",
        revision_one_inputs,
        cell_index=1,
    )
    advanced_revision = _site_map_protocol_view(
        _site_map_image(((21, 22), (23, 24))),
        "advanced-revision",
        _site_map_inputs(2),
        cell_index=1,
    )
    switched_advancing_revision = _site_map_protocol_view(
        _site_map_image(((31, 32), (33, 34))),
        "switched-advancing-revision",
        _site_map_inputs(3),
        cell_index=0,
    )
    geometry = panel_surface_geometry("2x2", pixel_ratio=1.25)
    moved_geometry = panel_surface_geometry("2x2", pixel_ratio=1.75)
    composer = site_map_render.SiteMapComposer(
        "sites",
        board_id="occupancy-cell",
        surface_geometry=geometry,
    )
    display = ImageDisplayState(relim_mode=RelimMode.NORMAL)
    try:
        composer.compose(first, display=display, selection_revision=3)
        composer.compose(first, display=display, selection_revision=3)
        composer.compose(
            second_frozen_cell,
            display=display,
            selection_revision=4,
        )
        composer.compose(
            advanced_revision,
            display=display,
            selection_revision=5,
        )
        composer.compose(
            switched_advancing_revision,
            display=display,
            selection_revision=6,
        )
        composer.compose(
            switched_advancing_revision,
            display=display,
            selection_revision=6,
            surface_geometry=moved_geometry,
            surface_revision=1,
        )
    finally:
        composer.close()

    assert range_inputs == [
        first.background,
        second_frozen_cell.background,
        advanced_revision.background,
        switched_advancing_revision.background,
    ]
    assert len(renderer_instances) == 2
    renderer, moved_renderer = renderer_instances
    assert (
        renderer.options["width"],
        renderer.options["height"],
    ) == geometry.raster_size
    assert renderer.options["dpi"] == geometry.dpi
    assert renderer.options["size_name"] == "2x2"
    assert renderer.options["site_map"] is True
    assert renderer.closed == 1
    assert (
        moved_renderer.options["width"],
        moved_renderer.options["height"],
    ) == moved_geometry.raster_size
    assert moved_renderer.options["dpi"] == moved_geometry.dpi
    assert moved_renderer.closed == 1
    assert compose_calls[0]["selection_revision"] == 3
    assert compose_calls[0]["current_color_limits"] is None
    assert compose_calls[0]["previous_relim_mode"] is None
    assert compose_calls[1]["current_color_limits"] == (2.0, 9.0)
    assert compose_calls[1]["previous_relim_mode"] is RelimMode.NORMAL
    assert compose_calls[2]["selection_revision"] == 4
    assert compose_calls[2]["current_color_limits"] is None
    assert compose_calls[2]["previous_relim_mode"] is None
    assert compose_calls[3]["selection_revision"] == 5
    assert compose_calls[3]["current_color_limits"] == (2.0, 9.0)
    assert compose_calls[3]["previous_relim_mode"] is RelimMode.NORMAL
    assert compose_calls[4]["selection_revision"] == 6
    assert compose_calls[4]["current_color_limits"] is None
    assert compose_calls[4]["previous_relim_mode"] is None
    assert compose_calls[5]["surface_revision"] == 1
    assert compose_calls[5]["current_color_limits"] == (2.0, 9.0)
    assert compose_calls[5]["previous_relim_mode"] is RelimMode.NORMAL


def test_occupancy_window_delegates_to_generic_figure_surface_and_closes():
    import zlc_neutral_atom.logic_nodes.readout.occupancy.ui.workbench_window as module
    from zlc_frontend.qt_widgets import (
        FigureSurfaceHost,
        FigureSurfaceLane,
        ensure_qt_app,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import (
        OccupancyArtifactRef,
    )

    application = ensure_qt_app()

    def invalid_navigation_loader(_reference):
        return object()

    window = module.OccupancyCellWindow(
        invalid_navigation_loader,
        lambda *_args, **_kwargs: None,
        OccupancyArtifactRef("test", "a" * 64),
        address=None,
    )
    try:
        deadline = time.monotonic() + 3.0
        while not window.worker_idle and time.monotonic() < deadline:
            application.processEvents()
        assert window.worker_idle
        assert window._status.text() == "OCCUPANCY CELL FAILED"
        assert isinstance(window._surface_host, FigureSurfaceHost)
        assert isinstance(window._surface_lane, FigureSurfaceLane)

        source = Path(module.__file__).read_text(encoding="utf-8")
        for retired_owner in (
            "SiteMapComposer",
            "SinglePanelHost",
            "RasterPixelRatioObserver",
            "FluentRevisionedFormEditor",
        ):
            assert retired_owner not in source

        window.shutdown()
        deadline = time.monotonic() + 3.0
        while not window.closed and time.monotonic() < deadline:
            application.processEvents()
        application.processEvents()
        assert window.closed
        assert window._surface_lane.shutdown_complete
    finally:
        if not window.closed:
            window.shutdown()
