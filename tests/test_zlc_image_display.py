from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from zlc_frontend.display_range import (
    RelimMode,
    deadband_display_range,
    target_display_range,
)


ROOT = Path(__file__).resolve().parents[1]


def test_image_display_owner_is_headless_closed_and_immutable() -> None:
    source = ROOT / "zlc_frontend" / "image_display.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    roots = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module.split(".", 1)[0]]
            if isinstance(node, ast.ImportFrom) and node.module
            else []
        )
    }
    assert roots.isdisjoint({"PyQt5", "matplotlib", "Zou_lab_control", "zlc_workbench"})

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from zlc_frontend.image_display import ImageDisplayState\n"
                "assert ImageDisplayState().revision == 0\n"
                "assert 'PyQt5' not in sys.modules\n"
                "assert 'matplotlib' not in sys.modules\n"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    from zlc_frontend.image_display import ImageDisplayState

    state = ImageDisplayState()
    with pytest.raises(FrozenInstanceError):
        state.revision = 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"revision": True}, TypeError),
        ({"revision": -1}, ValueError),
        ({"relim_mode": "tight"}, TypeError),
        ({"colormap": "gray"}, TypeError),
        ({"x_view": [0.0, 1.0]}, TypeError),
        ({"x_view": (0.0, float("nan"))}, ValueError),
        ({"x_view": (-1.7e308, 1.7e308)}, ValueError),
        ({"y_view": (2.0, 2.0)}, ValueError),
    ),
)
def test_image_display_state_rejects_ambiguous_or_nonfinite_values(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    from zlc_frontend.image_display import ImageDisplayState

    with pytest.raises(error):
        ImageDisplayState(**kwargs)


def test_fixed_state_requires_one_strict_color_range() -> None:
    from zlc_frontend.image_display import ImageDisplayState

    with pytest.raises(ValueError, match="requires fixed_color_limits"):
        ImageDisplayState(relim_mode=RelimMode.FIXED)
    state = ImageDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(10, 20),
    )
    assert state.fixed_color_limits == (10.0, 20.0)


def test_one_form_projection_is_exact_typed_and_revisioned() -> None:
    from zlc_frontend.image_display import (
        ImageColormap,
        ImageDisplayState,
        image_display_form_spec,
        image_display_form_values,
        image_display_from_form,
    )

    base = ImageDisplayState()
    spec = image_display_form_spec()
    assert spec.keys == (
        "relim_mode",
        "colormap",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "color_min",
        "color_max",
    )
    values = image_display_form_values(base)
    assert set(values) == set(spec.keys)
    assert values["relim_mode"] is RelimMode.TIGHT
    assert values["colormap"] is ImageColormap.GRAY
    assert image_display_from_form(base, values) is base

    values.update(
        colormap=ImageColormap.INFERNO,
        x_min=2,
        x_max=8,
        y_min=-3.5,
        y_max=4.5,
    )
    changed = image_display_from_form(base, values)
    assert changed.revision == 1
    assert changed.colormap is ImageColormap.INFERNO
    assert changed.x_view == (2.0, 8.0)
    assert changed.y_view == (-3.5, 4.5)

    with pytest.raises(ValueError, match="exactly"):
        image_display_from_form(changed, {**values, "extra": 1})
    with pytest.raises(TypeError, match="RelimMode"):
        image_display_from_form(changed, {**values, "relim_mode": "normal"})
    with pytest.raises(ValueError, match="both minimum and maximum"):
        image_display_from_form(changed, {**values, "x_max": None})


def test_entering_fixed_freezes_visible_range_then_accepts_explicit_edits() -> None:
    from zlc_frontend.image_display import (
        ImageDisplayState,
        image_display_form_values,
        image_display_from_form,
    )

    base = ImageDisplayState(fixed_color_limits=(1.0, 2.0))
    values = image_display_form_values(base)
    values["relim_mode"] = RelimMode.FIXED
    with pytest.raises(ValueError, match="current_color_limits"):
        image_display_from_form(base, values)

    fixed = image_display_from_form(
        base,
        values,
        current_color_limits=(20.0, 40.0),
    )
    assert fixed.revision == 1
    assert fixed.fixed_color_limits == (20.0, 40.0)

    edited_values = image_display_form_values(fixed)
    edited_values.update(color_min=25.0, color_max=35.0)
    edited = image_display_from_form(fixed, edited_values)
    assert edited.revision == 2
    assert edited.fixed_color_limits == (25.0, 35.0)


def test_target_color_limits_match_mature_tight_normal_and_fixed_rules() -> None:
    from zlc_frontend.image_display import ImageDisplayState

    tight = ImageDisplayState(relim_mode=RelimMode.TIGHT)
    normal = ImageDisplayState(relim_mode=RelimMode.NORMAL)
    fixed = ImageDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(11.0, 19.0),
    )
    assert target_display_range(tight.relim_mode, 50.0, 250.0) == (30.0, 270.0)
    assert target_display_range(normal.relim_mode, 50.0, 250.0) == (0.0, 300.0)
    assert target_display_range(normal.relim_mode, -50.0, 200.0) == (-75.0, 225.0)
    assert target_display_range(tight.relim_mode, 5.0, 5.0) == (4.5, 5.5)
    assert target_display_range(tight.relim_mode, 0.0, 0.0) == (-0.1, 0.1)
    assert target_display_range(normal.relim_mode, 0.0, 0.0) == (0.0, 1.0)
    assert target_display_range(
        fixed.relim_mode,
        float("nan"),
        float("nan"),
        fixed_range=fixed.fixed_color_limits,
    ) == (
        11.0,
        19.0,
    )


def test_deadband_never_clips_and_avoids_normal_or_tight_jitter() -> None:
    from zlc_frontend.image_display import ImageDisplayState

    normal = ImageDisplayState(relim_mode=RelimMode.NORMAL)
    assert deadband_display_range(normal.relim_mode, (0.0, 120.0), 0.0, 100.0) == (
        0.0,
        120.0,
    )
    assert deadband_display_range(normal.relim_mode, (0.0, 120.0), 0.0, 121.0) == (
        0.0,
        pytest.approx(145.2),
    )
    assert deadband_display_range(normal.relim_mode, (0.0, 120.0), 0.0, 50.0) == (
        0.0,
        60.0,
    )

    tight = ImageDisplayState(relim_mode=RelimMode.TIGHT)
    assert deadband_display_range(tight.relim_mode, (-1.0, 11.0), 0.1, 9.9) == (
        -1.0,
        11.0,
    )
    grown = deadband_display_range(tight.relim_mode, (-1.0, 11.0), 0.0, 20.0)
    assert grown == (-2.0, 22.0)
    assert grown[0] <= 0.0 and grown[1] >= 20.0
    assert deadband_display_range(tight.relim_mode, (-1.0, 11.0), 4.0, 6.0) == (
        3.8,
        6.2,
    )
    assert deadband_display_range(
        tight.relim_mode,
        (-1.0, 11.0),
        0.1,
        9.9,
        force=True,
    ) == pytest.approx((-0.88, 10.88))

    # A negative NORMAL frame derives TIGHT for this frame without mutating intent.
    negative = deadband_display_range(normal.relim_mode, (0.0, 120.0), -5.0, 100.0)
    assert negative == (-15.5, 110.5)
    assert normal.relim_mode is RelimMode.NORMAL

    # NORMAL must regain its zero anchor after a negative frame.  A high-value
    # deadband match alone cannot preserve the previous tight negative floor.
    positive_again = deadband_display_range(normal.relim_mode, negative, 1.0, 90.0)
    assert positive_again == (0.0, 108.0)
    negative_again = deadband_display_range(
        normal.relim_mode,
        positive_again,
        -2.0,
        80.0,
    )
    assert negative_again == pytest.approx((-10.2, 88.2))
    assert deadband_display_range(
        normal.relim_mode,
        negative_again,
        1.0,
        70.0,
    ) == (0.0, 84.0)


def _evaluated_image(values, validity):
    import numpy as np

    from zlc_data import AxisId, CoordinateFrameId, SPATIAL_X, SPATIAL_Y
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedImage

    array = np.asarray(values)
    height, width = array.shape
    frame = CoordinateFrameId("indexed-test.camera")
    return EvaluatedImage(
        EvaluatedAxis(
            AxisId("indexed-test.x"),
            "x",
            SPATIAL_X,
            "pixel",
            tuple(range(width)),
            tuple(range(width)),
            frame,
        ),
        EvaluatedAxis(
            AxisId("indexed-test.y"),
            "y",
            SPATIAL_Y,
            "pixel",
            tuple(range(height)),
            tuple(range(height)),
            frame,
        ),
        array,
        np.asarray(validity, dtype=bool),
        "photoelectron",
    )


def _rasterize(image, state=None):
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.image_raster import rasterize_image_indexed8

    return rasterize_image_indexed8(
        image,
        ImageDisplayState() if state is None else state,
        current_color_limits=None,
        previous_relim_mode=None,
    )


def test_evaluated_image_metadata_builds_one_exact_read_only_payload() -> None:
    from zlc_data import (
        BlockId,
        DatasetRevision,
        DatasetRevisionRef,
        StreamGenerationId,
    )
    from zlc_frontend.figure import DatasetId, EvaluatedInput
    from zlc_frontend.image_view import image_viewport_for_evaluated_image
    from zlc_frontend.render import ImagePanelPayload

    image = _evaluated_image(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[True, True, True], [True, True, True]],
    )
    viewport = image_viewport_for_evaluated_image(image)
    _raster, data_range, histogram, color_limits = _rasterize(image)
    payload = ImagePanelPayload(
        image=image,
        evaluated_input=EvaluatedInput(
            DatasetId("image-fidelity"),
            DatasetRevisionRef(
                BlockId("image-fidelity-block"),
                StreamGenerationId("image-fidelity-generation"),
                "a" * 64,
                DatasetRevision(1),
            ),
        ),
        viewport=viewport,
        data_range=data_range,
        histogram_counts=histogram,
        base_palette=tuple(range(256)),
        color_limits=color_limits,
    )

    assert viewport.x_axis.axis_id == image.x_axis.axis_id
    assert viewport.y_axis.axis_id == image.y_axis.axis_id
    assert viewport.x_axis.coordinates == image.x_axis.coordinates
    assert viewport.y_axis.coordinates == image.y_axis.coordinates
    assert viewport.coordinate_frame is image.x_axis.coordinate_frame
    assert payload.value_unit == "photoelectron"
    assert ImagePanelPayload.value_unit.fset is None


def test_evaluated_image_viewport_fails_closed_without_spatial_axis_truth() -> None:
    from dataclasses import replace

    from zlc_data import CoordinateFrameId, SCAN_POINT
    from zlc_frontend.figure import EvaluatedImage
    from zlc_frontend.image_view import image_viewport_for_evaluated_image

    image = _evaluated_image(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[True, True, True], [True, True, True]],
    )

    missing_frame = EvaluatedImage(
        replace(image.x_axis, coordinate_frame=None),
        image.y_axis,
        image.values,
        image.validity,
        image.value_unit,
    )
    with pytest.raises(ValueError, match="explicit coordinate frame"):
        image_viewport_for_evaluated_image(missing_frame)

    irregular = EvaluatedImage(
        replace(image.x_axis, coordinates=(0, 1, 3)),
        image.y_axis,
        image.values,
        image.validity,
        image.value_unit,
    )
    with pytest.raises(ValueError, match="not exactly regular"):
        image_viewport_for_evaluated_image(irregular)

    nonnumeric = EvaluatedImage(
        replace(image.x_axis, coordinates=("left", "middle", "right")),
        image.y_axis,
        image.values,
        image.validity,
        image.value_unit,
    )
    with pytest.raises(TypeError, match="finite numeric coordinates"):
        image_viewport_for_evaluated_image(nonnumeric)

    unnamed = EvaluatedImage(
        replace(image.x_axis, role=SCAN_POINT),
        image.y_axis,
        image.values,
        image.validity,
        image.value_unit,
    )
    with pytest.raises(ValueError, match="x_axis=SPATIAL_X"):
        image_viewport_for_evaluated_image(unnamed)

    mismatched_frame = EvaluatedImage(
        image.x_axis,
        replace(
            image.y_axis,
            coordinate_frame=CoordinateFrameId("another-camera"),
        ),
        image.values,
        image.validity,
        image.value_unit,
    )
    with pytest.raises(ValueError, match="shared, explicit coordinate frame"):
        image_viewport_for_evaluated_image(mismatched_frame)


def test_indexed_raster_reserves_zero_for_invalid_and_reports_exact_codebook() -> None:
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.render import PixelFormat

    image = _evaluated_image(
        [[0.0, 50.0, 100.0, float("nan")], [10.0, 20.0, 30.0, 40.0]],
        [[True, True, True, True], [False, True, True, True]],
    )
    state = ImageDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(0.0, 100.0),
    )
    raster, value_range, histogram, limits = _rasterize(image, state)

    assert raster.pixel_format is PixelFormat.INDEXED8
    assert raster.pixels == bytes((1, 128, 255, 0, 0, 51, 77, 102))
    assert value_range == (0.0, 100.0)
    assert limits == (0.0, 100.0)
    assert len(histogram) == 255 and sum(histogram) == 6
    assert histogram[0] == 1 and histogram[127] == 1 and histogram[254] == 1


def test_indexed_raster_handles_degenerate_or_empty_frames_without_fake_values() -> None:
    constant = _evaluated_image([[7, 7]], [[True, True]])
    raster, value_range, histogram, limits = _rasterize(constant)
    assert raster.pixels == bytes((128, 128))
    assert value_range == (7.0, 7.0)
    assert limits == pytest.approx((6.3, 7.7))
    assert histogram[127] == 2

    empty = _evaluated_image([[1.0, float("inf")]], [[False, True]])
    raster, value_range, histogram, limits = _rasterize(empty)
    assert raster.pixels == bytes((0, 0))
    assert value_range is None
    assert limits == (0.0, 1.0)
    assert sum(histogram) == 0


@pytest.mark.parametrize(
    "values",
    (
        [[0.0, 1.0e100]],
        [[-1.0e308, -5.0e307]],
    ),
)
def test_indexed_raster_maps_extreme_finite_float64_without_invalid_codes(values) -> None:
    image = _evaluated_image(values, [[True] * len(values[0])])
    raster, value_range, histogram, _limits = _rasterize(image)
    assert value_range == (float(min(values[0])), float(max(values[0])))
    assert all(code != 0 for code in raster.pixels)
    assert sum(histogram) == len(values[0])


def test_indexed_raster_rejects_an_unrepresentable_infinite_display_span() -> None:
    image = _evaluated_image(
        [[-1.0e308, 0.0, 1.0e308]],
        [[True, True, True]],
    )
    with pytest.raises(ValueError, match="span must be finite"):
        _rasterize(image)


def test_indexed_raster_maps_adjacent_subnormal_values_without_collapsing_validity() -> None:
    import numpy as np

    from zlc_frontend.image_display import ImageDisplayState

    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    image = _evaluated_image(
        np.asarray([[0.0, smallest, smallest * 2.0]], dtype=np.float64),
        [[True, True, True]],
    )
    state = ImageDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(0.0, float(smallest * 2.0)),
    )
    raster, value_range, histogram, _limits = _rasterize(image, state)
    assert value_range == (0.0, float(smallest * 2.0))
    assert raster.pixels[0] == 1 and raster.pixels[-1] == 255
    assert all(code != 0 for code in raster.pixels)
    assert sum(histogram) == 3


def test_display_raster_quantizes_directly_inside_narrow_fixed_limits() -> None:
    import numpy as np

    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.image_raster import rasterize_image_indexed8

    in_window = np.arange(1000, 1101, dtype=np.uint16)
    values = np.concatenate(
        (np.asarray([0], dtype=np.uint16), in_window, np.asarray([65535], dtype=np.uint16))
    ).reshape(1, -1)
    image = _evaluated_image(values, np.ones(values.shape, dtype=bool))
    state = ImageDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(1000.0, 1100.0),
    )
    raster, data_range, histogram, limits = rasterize_image_indexed8(
        image,
        state,
        current_color_limits=None,
        previous_relim_mode=None,
    )
    assert data_range == (0.0, 65535.0)
    assert limits == (1000.0, 1100.0)
    assert raster.pixels[0] == raster.pixels[1] == 1
    assert raster.pixels[-2] == raster.pixels[-1] == 255
    # A full-range-first 8-bit conversion leaves only about one source code in
    # this window.  Direct clim quantization preserves every integer level.
    assert len(set(raster.pixels[1:-1])) == len(in_window)
    assert sum(histogram) == len(in_window)


def test_indexed_raster_rejects_integer_ranges_float64_cannot_distinguish() -> None:
    import numpy as np

    base = np.uint64(2**63)
    image = _evaluated_image(
        np.asarray([[base, base + np.uint64(1)]], dtype=np.uint64),
        [[True, True]],
    )
    with pytest.raises(TypeError, match="explicit display transform"):
        _rasterize(image)


def test_indexed_memory_budget_counts_raster_and_exact_held_fronts_separately() -> None:
    import numpy as np

    from zlc_frontend.image_raster import estimate_indexed8_raster_peak_nbytes

    height, width = 17, 31
    one = estimate_indexed8_raster_peak_nbytes(
        height,
        width,
        value_itemsize=np.dtype(np.uint16).itemsize,
        retained_fronts=1,
        retained_sample_fronts=1,
    )
    held = estimate_indexed8_raster_peak_nbytes(
        height,
        width,
        value_itemsize=np.dtype(np.uint16).itemsize,
        retained_fronts=2,
        retained_sample_fronts=2,
    )
    assert held - one == height * width * (
        2 * np.dtype(np.uint8).itemsize
        + np.dtype(np.uint16).itemsize
        + np.dtype(bool).itemsize
    ) + 128 * (height + width)


@pytest.mark.parametrize("dtype", ("uint16", "float32"))
def test_indexed_raster_peak_stays_inside_single_workspace_budget(dtype: str) -> None:
    import gc
    import tracemalloc

    import numpy as np

    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.image_raster import (
        estimate_indexed8_raster_peak_nbytes,
        rasterize_image_indexed8,
    )

    height, width = 480, 640
    values = np.arange(height * width, dtype=np.dtype(dtype)).reshape(height, width)
    image = _evaluated_image(values, np.ones((height, width), dtype=bool))
    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    rasterize_image_indexed8(
        image,
        ImageDisplayState(),
        current_color_limits=None,
        previous_relim_mode=None,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    budget = estimate_indexed8_raster_peak_nbytes(
        height,
        width,
        value_itemsize=values.dtype.itemsize,
        retained_fronts=0,
        retained_sample_fronts=0,
    )
    assert peak <= budget


def _display_viewport(*, width: int = 8, height: int = 6):
    from zlc_data import (
        AxisId,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.image_view import ImageViewportTransform

    frame = CoordinateFrameId("image-display-viewport")
    return ImageViewportTransform(
        (
            AxisSpec(
                AxisId("image-display.y"),
                "y",
                SPATIAL_Y,
                height,
                tuple(100 - 3 * index for index in range(height)),
                unit="pixel",
                coordinate_frame=frame,
            ),
            AxisSpec(
                AxisId("image-display.x"),
                "x",
                SPATIAL_X,
                width,
                tuple(10 + 2 * index for index in range(width)),
                unit="pixel",
                coordinate_frame=frame,
            ),
        )
    )


def test_display_state_and_viewport_are_one_revisioned_view_truth() -> None:
    from zlc_frontend.image_display import (
        ImageDisplayState,
        image_display_for_viewport,
        image_viewport_for_display_state,
    )

    home = _display_viewport()
    base = ImageDisplayState()
    assert image_viewport_for_display_state(base, home) is home

    zoomed = home.centered_zoom((0.25, 0.75), 0.5, viewport_revision=1)
    authored = image_display_for_viewport(base, zoomed)
    assert authored.revision == 1
    assert authored.x_view is not None and authored.y_view is not None
    rebuilt = image_viewport_for_display_state(authored, home)
    assert rebuilt.viewport_revision == 1
    assert rebuilt == zoomed

    mismatched = ImageDisplayState(revision=1, x_view=(9.0, 25.0))
    with pytest.raises(ValueError, match="coordinate views differ"):
        image_viewport_for_display_state(mismatched, zoomed)


@pytest.mark.parametrize("descending", (False, True))
def test_viewport_coordinate_state_round_trip_is_exact_across_gestures(
    descending: bool,
) -> None:
    from zlc_data import (
        AxisId,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.image_display import (
        ImageDisplayState,
        image_display_for_viewport,
        image_viewport_for_display_state,
    )
    from zlc_frontend.image_view import ImageViewportTransform

    frame = CoordinateFrameId("large-offset-camera")
    x_coordinates = tuple(1_000_000_000 + 2 * index for index in range(257))
    y_coordinates = tuple(2_000_000_000 + 3 * index for index in range(193))
    if descending:
        x_coordinates = tuple(reversed(x_coordinates))
        y_coordinates = tuple(reversed(y_coordinates))
    home = ImageViewportTransform(
        (
            AxisSpec(
                AxisId("large-offset.x"),
                "x",
                SPATIAL_X,
                len(x_coordinates),
                x_coordinates,
                "pixel",
                frame,
            ),
            AxisSpec(
                AxisId("large-offset.y"),
                "y",
                SPATIAL_Y,
                len(y_coordinates),
                y_coordinates,
                "pixel",
                frame,
            ),
        )
    )
    state = ImageDisplayState()
    viewport = home

    for cycle in range(1, 33):
        if cycle % 2:
            pending = viewport.centered_zoom(
                (0.3141592653589793, 0.7182818284590452),
                0.97,
                viewport_revision=cycle,
            )
        else:
            pending = viewport.panned_by_pixels(
                (3.25, -2.75),
                (997, 613),
                viewport_revision=cycle,
            )
        state = image_display_for_viewport(state, pending)
        rebuilt = image_viewport_for_display_state(state, home)
        assert rebuilt == pending
        viewport = rebuilt


def test_unpinned_singleton_axis_never_needs_an_invented_coordinate_span() -> None:
    from zlc_frontend.image_display import (
        ImageDisplayState,
        image_display_for_viewport,
        image_viewport_for_display_state,
    )

    viewport = _display_viewport(width=1)
    state = ImageDisplayState()
    assert image_viewport_for_display_state(state, viewport) is viewport
    assert image_display_for_viewport(state, viewport) is state
