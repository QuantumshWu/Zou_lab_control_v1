"""Headless selection/fit engine contracts shared by plots and processors."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import tracemalloc

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.fitting import (
    FitRequest,
    HistogramFitResult,
    fit_data,
    fit_histogram,
    fit_image,
    fit_models,
)
from Zou_lab_control.neutral_atom.core.selection import Selection, select_rows


def test_small_selection_fails_instead_of_silently_fitting_everything():
    x = np.linspace(-10, 10, 101)
    y = np.exp(-x**2)
    request = FitRequest("gaussian", selection=Selection.x(9.9, 10.0))
    result = fit_data({"x": x}, y, request)
    assert result.valid is False
    assert "selection contains" in result.status
    assert np.isnan(result.parameters).all()


def test_one_dimensional_fit_uses_the_same_selected_range_semantics():
    x = np.linspace(-10, 10, 801)
    # Two peaks: the selected left range must fit x0=-3, not the larger right peak.
    y = (2.0 * np.exp(-((x + 3.0) ** 2) / (2 * 0.8**2))
         + 5.0 * np.exp(-((x - 4.0) ** 2) / (2 * 0.9**2)) + 0.2)
    result = fit_data({"x": x}, y, FitRequest("gaussian", selection=Selection.x(-6, 0)))
    assert result.valid, result.status
    assert result.parameter_map()["x0"] == pytest.approx(-3.0, abs=0.05)
    assert result.quality["r2"] > 0.999


def test_two_dimensional_center_fit_reports_coordinate_frame_and_quality():
    yy, xx = np.mgrid[0:160, 0:220]
    image = 12.0 + 90.0 * np.exp(-((xx - 137.5) ** 2 + (yy - 66.25) ** 2) / 18.0**2)
    selection = Selection.rectangle(100, 175, 30, 105, frame="camera_px")
    result = fit_data(
        {"x": xx.ravel(), "y": yy.ravel()}, image.ravel(),
        FitRequest("center", selection=selection, coordinate_frame="camera_px", sample_budget=6000),
    )
    assert result.valid, result.status
    params = result.parameter_map()
    assert params["x0"] == pytest.approx(137.5, abs=0.25)
    assert params["y0"] == pytest.approx(66.25, abs=0.25)
    assert result.coordinate_frame == "camera_px"
    assert result.quality["r2"] > 0.999


def test_fit_image_matches_full_coordinates_after_selection_sampling_and_nans():
    yy, xx = np.mgrid[0:90, 0:130]
    origin = (100.25, -40.5)
    image = 4.0 + 30.0 * np.exp(-((xx - 72.5) ** 2 + (yy - 41.25) ** 2) / 11.0**2)
    image[::13, ::11] = np.nan
    selection = Selection.rectangle(145.0, 205.0, -25.0, 25.0, frame="camera_px")
    request = FitRequest(
        "center", selection=selection, coordinate_frame="camera_px",
        sample_budget=700, solve=False,
    )

    raster = fit_image(image, request, origin=origin)
    rows = fit_data(
        {"x": (xx + origin[0]).ravel(), "y": (yy + origin[1]).ravel()},
        image.ravel(), request,
    )

    assert raster.valid and rows.valid
    assert raster.n_points == rows.n_points == request.sample_budget
    np.testing.assert_allclose(raster.parameters, rows.parameters, rtol=0.0, atol=1e-12)


def test_fit_image_budget_does_not_copy_full_frame_or_all_valid_indices():
    # Allocate the source before tracing: the contract measures incremental fit
    # workspace, not ownership of the caller's image.
    image = np.tile(np.linspace(0.0, 1.0, 1920), (1200, 1))
    request = FitRequest("center", sample_budget=12_000, solve=False)

    tracemalloc.start()
    result = fit_image(image, request)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.valid, result.status
    assert result.n_points == request.sample_budget
    # One finite bool mask plus O(height + budget) indices is expected.  Either
    # the old selected-frame copy or the old int64 flatnonzero array alone would
    # exceed half of this float64 frame.
    assert peak < image.nbytes // 2, f"fit_image used {peak:,} B for an {image.nbytes:,} B frame"


def test_selection_is_vectorized_and_missing_axis_is_loud():
    source = inspect.getsource(Selection.mask)
    assert "enumerate(" not in source and "for i in" not in source
    with pytest.raises(KeyError, match="coordinate 'y'"):
        select_rows({"x": np.arange(5)}, np.arange(5), Selection.rectangle(0, 4, 0, 1))


def test_grid_scope_round_trips_without_becoming_global():
    selection = Selection.rectangle(1, 2, 3, 4, scope=("facet:repeat", "cell:7"))
    assert Selection.from_dict(selection.to_dict()) == selection
    assert selection.scope == ("facet:repeat", "cell:7")


def test_fit_rejects_multicomponent_y_without_explicit_selection_or_reduction():
    x = np.linspace(-1.0, 1.0, 21)
    values = np.column_stack((np.exp(-(x**2)), np.exp(-((x - 0.2) ** 2))))

    result = fit_data({"x": x}, values, FitRequest("gaussian"))

    assert not result.valid
    assert "exactly one scalar value" in result.status
    assert "Select a data component or reduce" in result.status


def test_model_catalogue_is_an_immutable_family_query():
    assert tuple(model.key for model in fit_models(family="2D")) == ("center",)
    assert tuple(model.key for model in fit_models(family="1d")) == (
        "lorent", "gaussian", "lorent_zeeman", "rabi", "decay")
    assert isinstance(fit_models(), tuple)


def test_histogram_fit_is_typed_and_owns_model_evaluation():
    edges = np.linspace(-8.0, 12.0, 81)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = (
        180.0 * np.exp(-((centers + 2.0) ** 2) / (2.0 * 0.8**2))
        + 140.0 * np.exp(-((centers - 5.0) ** 2) / (2.0 * 1.1**2))
    )

    result = fit_histogram(edges, counts, "double")

    assert isinstance(result, HistogramFitResult)
    assert result.valid and result.component_count == 2
    assert result.bimodal_parameters is not None
    assert result.single_parameters is None
    assert result.separated and -2.0 < result.threshold < 5.0
    left, right, total = result.curves(centers)
    assert left is not None and right is not None
    np.testing.assert_allclose(total, left + right)


def test_frontend_contains_no_fit_solver_or_model_registry_copy():
    import Zou_lab_control.frontend.live as live

    tree = ast.parse(Path(live.__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "scipy.optimize"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "curve_fit"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_histogram_curves"
        for node in ast.walk(tree)
    )
