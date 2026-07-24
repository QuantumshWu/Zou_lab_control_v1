"""Unified Figure Fit product and authority contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.workbench import open_figure_workbench
from zlc_workbench.data_figure.window import DataFigureWindow
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    fit_model_catalog,
    suggest_fit_draft,
)
from zlc_frontend import DataFigure, ImagePanelPayload, fit_authoring_option
from zlc_frontend.figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    suggest_view,
)
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_neutral_atom.artifacts import (
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
)


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "pulses" / "probe_template.json"
PANEL_ID = "generic-typed"


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture(scope="module")
def capture_product(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("u03d-unified-fit")
    with zlc.connect("virtual", repository=workspace) as experiment:
        reference = experiment.readout.capture(PULSE)
        yield experiment, reference, workspace


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _close(application, window) -> None:
    window.close()
    _until(application, lambda: window.closed and not window.isVisible(), timeout=10.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _board(window: DataFigureWindow) -> QtRasterBoard:
    board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
    assert board is not None and board.front_frame is not None
    return board


def _image_payload(window: DataFigureWindow) -> ImagePanelPayload:
    payload = _board(window).visible_image_payload(PANEL_ID)
    assert isinstance(payload, ImagePanelPayload)
    return payload


def _image_target(board: QtRasterBoard):
    binding = board._image_bindings[PANEL_ID]
    target = board._selector_target(binding)
    assert target is not None
    return binding, target[0]


def _point(target, x_fraction: float, y_fraction: float) -> QtCore.QPoint:
    return QtCore.QPoint(
        target.left() + int(x_fraction * max(1, target.width() - 1)),
        target.top() + int(y_fraction * max(1, target.height() - 1)),
    )


def _drag_move(board: QtRasterBoard, position: QtCore.QPoint, button) -> None:
    board.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(position),
            QtCore.Qt.NoButton,
            button,
            QtCore.Qt.NoModifier,
        )
    )


def _wheel(board: QtRasterBoard, position: QtCore.QPoint, delta: int) -> None:
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    board.wheelEvent(event)
    assert event.isAccepted()


def _drag_image_roi(window: DataFigureWindow) -> Selection:
    board = _board(window)
    _binding, target = _image_target(board)
    start = _point(target, 0.18, 0.2)
    end = _point(target, 0.82, 0.78)
    QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
    _drag_move(board, end, QtCore.Qt.LeftButton)
    QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
    candidate = window._fit_candidate
    assert candidate is not None
    return candidate.selection


def _open_image_fit(application, experiment, reference) -> DataFigureWindow:
    window = experiment.figure_gui(reference)
    assert isinstance(window, DataFigureWindow)
    _until(application, lambda: window.worker_idle and window.raster_ready)
    assert window._view_family == "image"
    assert window._fit_pane is not None
    assert window._tabs.indexOf(window._fit_pane) < 0
    assert window._fit_button.isEnabled()
    QtTest.QTest.mouseClick(window._fit_button, QtCore.Qt.LeftButton)
    _until(application, lambda: window.worker_idle and bool(window.fit_models))
    return window


def _curve_figure() -> tuple[DataFigure, DatasetSchema, AxisSpec, AxisSpec]:
    repeat = AxisSpec(AxisId("u03d.repeat"), "repeat", REPEAT, 1, (0,))
    scan = AxisSpec(
        AxisId("u03d.detuning"),
        "detuning",
        SCAN_POINT,
        21,
        tuple(float(index) for index in range(21)),
        "MHz",
        None,
    )
    site = AxisSpec(AxisId("u03d.site"), "site", SITE, 1, ("only",))
    x = np.asarray(scan.coordinates, dtype=np.float64)
    values = (2.0 + 8.0 * np.exp(-((x - 10.0) / 3.0) ** 2))[None, :, None]
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((site,), ValidityContract.value(), values.dtype, "count"),
    )
    block = DataBlock(
        BlockId("u03d-curve"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03d-curve-generation")),
        block,
    )
    suggestion = suggest_view(schema, ViewIntent.CURVE)
    assert suggestion.spec is not None
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "u03d-curve-document",
        0,
        (DatasetDescriptor(dataset_id, "curve", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    figure = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )
    return figure, schema, scan, site


def test_public_fit_surfaces_remain_headless_until_a_window_is_opened() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import Zou_lab_control.notebook; "
                "import Zou_lab_control.workbench; import zlc_frontend; "
                "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stderr


def test_figure_fit_is_one_step_save_reopen_refit_and_export(
    application,
    capture_product,
    tmp_path,
    monkeypatch,
) -> None:
    experiment, reference, _workspace = capture_product
    owner_thread = threading.get_ident()
    execute_threads: list[int] = []
    original_execute = FitResultRepository.execute_capture

    def observed_execute(self, *args, **kwargs):
        execute_threads.append(threading.get_ident())
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(FitResultRepository, "execute_capture", observed_execute)
    window = _open_image_fit(application, experiment, reference)
    try:
        pane = window._fit_pane
        assert pane is not None
        assert pane.model_combo.currentData() == "radial_gaussian_center"
        bound = pane.current_option()
        assert bound.fit_axis_roles == (SPATIAL_X, SPATIAL_Y)
        assert bound.spec.committed_transform is None
        assert not window.draft_ready and window.saved_reference is None

        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.worker_idle and window.draft_ready and window.raster_ready,
        )
        assert execute_threads and all(item != owner_thread for item in execute_threads)
        payload = _image_payload(window)
        assert payload.fit_overlay is not None
        assert payload.fit_overlay.result_identity.startswith("draft-fit:")
        assert pane.save_button.isEnabled()

        destination = tmp_path / "image-fit-overlay.png"
        window._start_export(destination)
        _until(application, lambda: window.worker_idle and destination.exists())
        assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

        QtTest.QTest.mouseClick(pane.save_button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.worker_idle
            and window.saved_reference is not None
            and window.raster_ready,
        )
        saved = window.saved_reference
        assert isinstance(saved, FitResultArtifactRef)
        admitted = experiment.load_fit(saved)
        assert admitted.source_artifact_ref == reference
        assert _image_payload(window).fit_overlay.result_identity == (
            f"{saved.repository_id}:{saved.manifest_digest}"
        )

        # Refit is a new explicit draft; the immutable saved reference remains.
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(application, lambda: window.worker_idle and window.draft_ready)
        assert window.saved_reference == saved
        QtTest.QTest.mouseClick(pane.clear_button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.worker_idle
            and not window.draft_ready
            and _image_payload(window).fit_overlay is None,
        )
        assert window.saved_reference == saved
    finally:
        _close(application, window)
    assert window._fit_draft is None
    assert window._fit_save_inflight is None
    assert window._fit_options == {}


def test_image_box_is_authority_and_use_full_range_removes_it(
    application,
    capture_product,
) -> None:
    experiment, reference, _workspace = capture_product
    window = experiment.fit_gui(reference, model="radial_gaussian_center")
    try:
        assert isinstance(window, DataFigureWindow)
        _until(
            application,
            lambda: window.worker_idle and window.raster_ready and bool(window.fit_models),
        )
        pane = window._fit_pane
        assert pane is not None and window._tabs.currentWidget() is pane
        selection = _drag_image_roi(window)
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        bound = pane.current_option()
        assert bound.spec.committed_transform is not None
        assert tuple(bound.spec.committed_transform.spec.operations) == (selection,)
        assert pane.full_range_button.isEnabled()
        assert "AUTHORITATIVE" in pane.authority_summary.text()

        QtTest.QTest.mouseClick(pane.full_range_button, QtCore.Qt.LeftButton)
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        assert window._fit_candidate is None
        assert pane.current_option().spec.committed_transform is None
        assert not pane.full_range_button.isEnabled()
        assert _board(window)._image_bindings[PANEL_ID].applied_bounds is None
    finally:
        _close(application, window)


def test_curve_range_promotes_only_x_while_display_cell_stays_presentation(
    application,
) -> None:
    figure, schema, x_axis, batch_axis = _curve_figure()

    def prepare(fit_axis_ids, authority_selection):
        return tuple(
            fit_authoring_option(
                suggest_fit_draft(
                    schema,
                    model.model_id,
                    fit_axis_ids=tuple(fit_axis_ids),
                    selection=authority_selection,
                )
            )
            for model in fit_model_catalog()
            if len(model.axis_requirements) == 1
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("curve preparation must not execute or save")

    window = open_figure_workbench(
        lambda _source, **_options: figure,
        "frozen-curve",
        fit_preparer=prepare,
        fit_executor=forbidden,
        fit_saver=forbidden,
        fit_reloader=forbidden,
    )
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert window._view_family == "curve"
        QtTest.QTest.mouseClick(window._fit_button, QtCore.Qt.LeftButton)
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        pane = window._fit_pane
        assert pane is not None
        model_index = pane.model_combo.findData("exponential_decay")
        assert model_index >= 0
        pane.model_combo.setCurrentIndex(model_index)

        board = _board(window)
        binding = board._numeric_binding_for_kind("curve", panel_id=PANEL_ID)
        assert binding is not None
        target = board._numeric_target(binding)
        assert target is not None
        start = QtCore.QPoint(
            int(round(target.plot.left() + 0.25 * target.plot.width())),
            int(round(target.plot.center().y())),
        )
        end = QtCore.QPoint(
            int(round(target.plot.left() + 0.85 * target.plot.width())),
            int(round(target.plot.center().y())),
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        bound = pane.current_option()
        transform = bound.spec.committed_transform
        assert transform is not None
        authority = transform.spec.operations
        assert len(authority) == 1 and isinstance(authority[0], Selection)
        assert tuple(term.axis_id for term in authority[0].terms) == (x_axis.axis_id,)
        assert batch_axis.axis_id in bound.spec.batch_axis_ids

        QtTest.QTest.mouseClick(pane.full_range_button, QtCore.Qt.LeftButton)
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        assert pane.current_option().spec.committed_transform is None
    finally:
        _close(application, window)


def test_fit_solver_does_not_block_image_navigation_and_new_roi_revokes_it(
    application,
    capture_product,
    monkeypatch,
) -> None:
    experiment, reference, _workspace = capture_product
    entered = threading.Event()
    release = threading.Event()
    original_execute = FitResultRepository.execute_capture

    def blocked_execute(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release Fit execution")
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(FitResultRepository, "execute_capture", blocked_execute)
    window = _open_image_fit(application, experiment, reference)
    try:
        pane = window._fit_pane
        assert pane is not None
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(application, entered.is_set)
        assert window._fit_future is not None

        board = _board(window)
        _binding, target = _image_target(board)
        _wheel(board, target.center(), -120)
        _until(application, lambda: window.raster_ready and window._display.revision == 1)
        assert window._fit_future is not None

        _drag_image_roi(window)
        release.set()
        _until(application, lambda: window.worker_idle)
        assert not window.draft_ready
        assert _image_payload(window).fit_overlay is None
        assert window._fit_candidate is not None
    finally:
        release.set()
        _close(application, window)


def test_fit_overlay_render_blocks_duplicate_fit_and_save_submission(
    application,
    capture_product,
) -> None:
    experiment, reference, _workspace = capture_product
    entered = threading.Event()
    release = threading.Event()
    window = _open_image_fit(application, experiment, reference)
    original_renderer = window._fit_overlay_renderer
    assert original_renderer is not None

    def blocked_overlay(*args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release Fit overlay rendering")
        return original_renderer(*args, **kwargs)

    window._fit_overlay_renderer = blocked_overlay
    try:
        pane = window._fit_pane
        assert pane is not None
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(application, entered.is_set)
        assert window._fit_future is None
        assert window._future is not None
        assert window._active_kind == "fit_overlay"
        assert window.draft_ready
        assert not pane.fit_button.isEnabled()
        assert not pane.save_button.isEnabled()

        release.set()
        _until(
            application,
            lambda: window.worker_idle and window.raster_ready and window.draft_ready,
        )
        assert _image_payload(window).fit_overlay is not None
        assert pane.fit_button.isEnabled()
        assert pane.save_button.isEnabled()
    finally:
        release.set()
        _close(application, window)


def test_viewport_rerender_rejects_a_self_consistent_foreign_fit_identity(
    application,
    capture_product,
) -> None:
    experiment, reference, _workspace = capture_product
    window = _open_image_fit(application, experiment, reference)
    try:
        pane = window._fit_pane
        assert pane is not None
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.worker_idle and window.draft_ready and window.raster_ready,
        )
        visible = _image_payload(window).fit_overlay
        assert visible is not None
        original_frame = _board(window).front_frame
        assert window._display is not None
        original_display_revision = window._display.revision
        original_renderer = window._typed_renderer

        def foreign_identity_renderer(
            fit_result,
            fit_result_identity,
            *args,
        ):
            assert fit_result is not None and fit_result_identity is not None
            return original_renderer(
                fit_result,
                f"foreign:{fit_result_identity}",
                *args,
            )

        window._typed_renderer = foreign_identity_renderer
        board = _board(window)
        _binding, target = _image_target(board)
        _wheel(board, target.center(), -120)
        _until(application, lambda: window.worker_idle)

        assert board.front_frame is original_frame
        assert (
            window._display is not None
            and window._display.revision == original_display_revision
        )
        assert "another Fit result identity" in window._diagnostic.text()
        admitted = _image_payload(window).fit_overlay
        assert admitted is not None
        assert admitted.result_identity == visible.result_identity
    finally:
        _close(application, window)


def test_failed_save_cannot_restore_draft_after_selector_revision_changes(
    application,
    capture_product,
    monkeypatch,
) -> None:
    experiment, reference, _workspace = capture_product
    entered = threading.Event()
    release = threading.Event()

    def failed_save(self):
        del self
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release failed Fit save")
        raise OSError("synthetic durable publication failure")

    monkeypatch.setattr(FitExecution, "save", failed_save)
    window = _open_image_fit(application, experiment, reference)
    try:
        pane = window._fit_pane
        assert pane is not None
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.worker_idle and window.draft_ready and window.raster_ready,
        )
        QtTest.QTest.mouseClick(pane.save_button, QtCore.Qt.LeftButton)
        _until(application, entered.is_set)
        assert window._fit_save_inflight is not None

        selection = _drag_image_roi(window)
        release.set()
        _until(application, lambda: window.worker_idle and bool(window.fit_models))
        assert window.saved_reference is None
        assert window._fit_save_inflight is None
        assert not window.draft_ready
        assert _image_payload(window).fit_overlay is None
        assert window._fit_candidate is not None
        assert window._fit_candidate.selection == selection
        bound = pane.current_option()
        assert bound.spec.committed_transform is not None
        assert tuple(bound.spec.committed_transform.spec.operations) == (selection,)
        assert not pane.save_button.isEnabled()
    finally:
        release.set()
        _close(application, window)


def test_close_during_atomic_save_accepts_reference_then_releases_heavy_state(
    application,
    capture_product,
    monkeypatch,
) -> None:
    experiment, reference, _workspace = capture_product
    published = threading.Event()
    release = threading.Event()
    original_save = FitExecution.save

    def blocked_save(self):
        saved = original_save(self)
        published.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release saved reference")
        return saved

    monkeypatch.setattr(FitExecution, "save", blocked_save)
    window = experiment.fit_gui(reference, model="radial_gaussian_center")
    try:
        _until(
            application,
            lambda: window.worker_idle and window.raster_ready and bool(window.fit_models),
        )
        pane = window._fit_pane
        assert pane is not None
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _until(application, lambda: window.worker_idle and window.draft_ready)
        QtTest.QTest.mouseClick(pane.save_button, QtCore.Qt.LeftButton)
        _until(application, published.is_set)
        window.close()
        application.processEvents()
        assert not window.closed
        assert "CLOSE DEFERRED" in window._status.text()
        release.set()
        _until(application, lambda: window.closed and window.saved_reference is not None)
        assert window._fit_draft is None
        assert window._fit_save_inflight is None
        assert window._fit_options == {}
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
