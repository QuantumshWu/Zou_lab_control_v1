"""W7 exact saved-fit GridPlot, sparse topology, and reopen oracles."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayout,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_result_retained_upper_bound_nbytes,
    fit_spec_for,
)
from zlc_data.fit_model import evaluate_fit_model
from zlc_frontend import DataFigure, FitGridModel
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    RepeatViewMode,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    suggest_fit_view,
)
from zlc_frontend.qt_widgets import AxisLayoutNavigator
from zlc_neutral_atom.artifacts import AdmittedCapture, CaptureFitResultRepository


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.close()
    _until(
        application,
        lambda: window.closed and not window.isVisible(),
        timeout=10.0,
    )


def _axis(name, role, size, coordinates=None) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        None,
        None,
    )


@pytest.fixture(scope="module")
def sparse_fit_grid():
    repeat = _axis("repeat", REPEAT, 2)
    event = _axis("event", SCAN_POINT, 3)
    y_axis = _axis("camera.y", SPATIAL_Y, 6)
    x_axis = _axis("camera.x", SPATIAL_X, 8)
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
        BlockId("w7-sparse-fit"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("w7-sparse-generation")),
        block,
    )
    result = bind_fit(
        fit_spec_for(schema, "radial_gaussian_center"),
        schema,
    ).run(snapshot)
    model = FitGridModel.from_result("fit-result/" + "f" * 64, result)
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
        "w7-sparse-grid",
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
    return result, model, page, suggestion.spec, figure


@pytest.fixture(scope="module")
def saved_fit_product(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("w7-saved-fit-workspace")
    with zlc.connect("virtual", repository=workspace) as experiment:
        capture = experiment.readout.capture(PULSE)
        reference = experiment.fit(
            capture,
            model="radial_gaussian_center",
        ).save()
        yield experiment, reference, workspace


def _manifest_count(workspace: Path) -> int:
    root = workspace / "fits" / "content" / "manifests" / "fit-result"
    return 0 if not root.exists() else len(tuple(root.iterdir()))


def test_saved_fit_grid_public_imports_stay_headless_and_ref_has_exact_identity():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zlc_frontend; import Zou_lab_control.notebook; "
                "import Zou_lab_control.workbench; "
                "from zlc_neutral_atom.capture_fit_reference import "
                "CaptureFitResultArtifactRef; "
                "r=CaptureFitResultArtifactRef('repo','f'*64); "
                "assert r.target_ref == 'fit-result/' + 'f'*64; "
                "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') "
                "for n in sys.modules); "
                "assert not any(n == 'matplotlib' or n.startswith('matplotlib.') "
                "for n in sys.modules); "
                "assert not any(n == 'scipy' or n.startswith('scipy.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stderr


def test_repeat_is_a_bounded_facet_and_sparse_holes_keep_logical_positions(
    sparse_fit_grid,
):
    result, model, page, view, figure = sparse_fit_grid
    repeat_axis = next(axis for axis in model.axes if axis.role == REPEAT)
    assert page.preferences.repeat_mode is RepeatViewMode.FACET
    assert view.binding(repeat_axis.axis_id).role is AxisViewRole.FACET
    assert all(
        view.binding(axis.axis_id).role is AxisViewRole.FACET
        for axis in model.axes
    )
    assert not hasattr(model, "result")
    assert model.layout.storage_size == result.batch_layout.storage_size == 4
    assert fit_result_retained_upper_bound_nbytes(result) > sum(
        array.nbytes
        for array in (
            result.parameter_values,
            result.covariance,
            result.residual_sum_squares,
        )
    )

    _payload, regions = figure.to_png_bytes_with_panel_regions(
        memory_limit_bytes=200 << 20,
    )
    assert len(regions) == 6
    present = tuple(region for region in regions if region.fit_storage_index is not None)
    holes = tuple(region for region in regions if region.fit_storage_index is None)
    assert len(present) == 4
    assert len(holes) == 2
    assert {region.fit_storage_index for region in present} == set(range(4))
    for region in regions:
        assert model.storage_index_or_none(region.selection) == region.fit_storage_index


def test_grid_pages_are_bounded_and_axis_navigator_skips_sparse_holes(
    application,
    sparse_fit_grid,
):
    _result, model, _page, _view, _figure = sparse_fit_grid
    navigator = AxisLayoutNavigator(
        model.axes,
        model.layout,
        object_prefix="w7Oracle",
        action_text="Focus",
    )
    try:
        navigator.set_storage_index(0)
        assert navigator.indices == model.layout.multi_index(0)
        navigator.next_button.click()
        assert navigator.indices == model.layout.multi_index(1)
        assert navigator.storage_index == 1
        hole = (0, 1)
        for (_axis_value, spin, _coordinate), index in zip(
            navigator._controls,
            hole,
            strict=True,
        ):
            spin.setValue(index)
        assert navigator.storage_index is None
        assert not navigator.action_button.isEnabled()
    finally:
        navigator.deleteLater()
        application.processEvents()


def test_large_grid_is_tiled_without_flattening_or_defaulting_repeat() -> None:
    repeat = _axis("large.repeat", REPEAT, 80)
    y_axis = _axis("large.y", SPATIAL_Y, 3)
    x_axis = _axis("large.x", SPATIAL_X, 3)
    model = FitGridModel(
        "fit-result/" + "e" * 64,
        "radial_gaussian_center",
        (x_axis, y_axis),
        (repeat,),
        AxisLayout.rect_c((repeat.size,)),
        ((FitBatchStatus.CONVERGED, repeat.size),),
    )
    first = model.page()
    second = model.page(first.next_address)
    third = model.page(second.next_address)
    assert model.page_spans == (36,)
    assert first.label == "large.repeat[0:36]"
    assert second.label == "large.repeat[36:72]"
    assert third.label == "large.repeat[72:80]"
    assert first.preferences.repeat_mode is RepeatViewMode.FACET
    assert first.previous_address is None
    assert third.next_address is None
    assert all(
        term.stop - term.start <= 36
        for page in (first, second, third)
        for term in page.selection.terms
    )


def test_saved_ref_dispatch_reopens_once_per_view_without_refit_and_exports(
    application,
    saved_fit_product,
    monkeypatch,
    tmp_path,
):
    experiment, reference, workspace = saved_fit_product
    owner_thread = threading.get_ident()
    load_threads = []
    materialize_threads = []
    original_load = CaptureFitResultRepository.load
    original_materialize = AdmittedCapture.materialize_snapshot

    def observed_load(self, *args, **kwargs):
        load_threads.append(threading.get_ident())
        return original_load(self, *args, **kwargs)

    def forbidden_execute(*_args, **_kwargs):
        raise AssertionError("saved-fit reopen/export must never run the solver")

    def observed_materialize(self, *args, **kwargs):
        materialize_threads.append(threading.get_ident())
        return original_materialize(self, *args, **kwargs)

    monkeypatch.setattr(CaptureFitResultRepository, "load", observed_load)
    monkeypatch.setattr(CaptureFitResultRepository, "execute", forbidden_execute)
    monkeypatch.setattr(AdmittedCapture, "materialize_snapshot", observed_materialize)
    manifests_before = _manifest_count(workspace)
    window = experiment.figure_gui(reference)
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert type(window).__name__ == "SavedFitGridWindow"
        assert load_threads == [load_threads[0]]
        assert load_threads[0] != owner_thread
        assert materialize_threads == [load_threads[0]]
        assert not hasattr(window._model, "result")
        assert window._page is not None
        assert len(window._regions) <= 36
        assert _manifest_count(workspace) == manifests_before

        region = next(
            item for item in window._regions if item.fit_storage_index is not None
        )
        window._focus_at(
            (region.left + region.right) / 2.0,
            (region.top + region.bottom) / 2.0,
        )
        _until(
            application,
            lambda: window.worker_idle and window._current_selection == region.selection,
        )
        assert len(load_threads) == 1
        assert len(materialize_threads) == 1
        storage, multi, _label = window._model.resolve_selection(region.selection)
        assert storage == region.fit_storage_index
        assert window._navigator.indices == multi
        assert f"storage row {storage}" in window._cell_detail.text()
        assert "status " in window._cell_detail.text()

        window._show_page()
        assert window._current_selection is None
        destination = tmp_path / "saved-fit-grid.png"
        window._start_export(destination)
        _until(
            application,
            lambda: window.worker_idle and destination.exists(),
        )
        assert destination.stat().st_size > 0
        assert len(load_threads) == 1
        assert len(materialize_threads) == 1
        assert all(thread != owner_thread for thread in load_threads)
        assert _manifest_count(workspace) == manifests_before
    finally:
        _close(application, window)


def test_saved_fit_grid_budget_and_close_are_fail_closed(
    application,
    saved_fit_product,
    monkeypatch,
):
    experiment, reference, _workspace = saved_fit_product

    import zlc_neutral_atom.artifacts.capture_fit as capture_fit_module

    with monkeypatch.context() as budget_patch:
        original_load = CaptureFitResultRepository.load

        def budgeted_load(self, *args, **kwargs):
            assert kwargs["memory_limit_bytes"] == 1
            return original_load(self, *args, **kwargs)

        def forbidden_decode(*_args, **_kwargs):
            raise AssertionError("tiny-budget load reached fit-result decode")

        budget_patch.setattr(CaptureFitResultRepository, "load", budgeted_load)
        budget_patch.setattr(
            capture_fit_module,
            "decode_fit_result_batch",
            forbidden_decode,
        )
        tiny = experiment.figure_gui(reference, memory_limit_bytes=1)
        try:
            _until(application, lambda: tiny.worker_idle)
            assert not tiny.raster_ready
            assert tiny._model is None
            assert tiny._status.text() == "SAVED FIT GRID FAILED"
            assert "MemoryError" in tiny._diagnostic.text()
        finally:
            _close(application, tiny)

    entered = threading.Event()
    release = threading.Event()
    original_load = CaptureFitResultRepository.load

    def blocked_load(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release saved-fit load")
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(CaptureFitResultRepository, "load", blocked_load)
    window = experiment.figure_gui(reference)
    try:
        _until(application, entered.is_set)
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < 0.1
        assert not window.closed
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert window._model is None
        assert window._page_bundle is None
        assert window._regions == ()
    finally:
        release.set()
        if not window.closed:
            _close(application, window)


def test_close_during_export_preserves_existing_destination_atomically(
    application,
    saved_fit_product,
    monkeypatch,
    tmp_path,
):
    experiment, reference, _workspace = saved_fit_product
    window = experiment.figure_gui(reference)
    _until(application, lambda: window.worker_idle and window.raster_ready)

    destination = tmp_path / "saved-fit-grid.png"
    original_bytes = b"pre-existing-authoritative-destination"
    destination.write_bytes(original_bytes)
    entered = threading.Event()
    release = threading.Event()
    original_export = DataFigure.export

    def blocked_export(self, *args, **kwargs):
        exported = original_export(self, *args, **kwargs)
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release staged fit export")
        return exported

    monkeypatch.setattr(DataFigure, "export", blocked_export)
    try:
        window._start_export(destination)
        _until(application, entered.is_set)
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < 0.1
        assert not window.closed
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert destination.read_bytes() == original_bytes
        assert not tuple(tmp_path.glob(f".{destination.name}.*"))
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
