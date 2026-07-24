"""Product contracts for the typed generic no-fit IMAGE DataFigure window."""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import threading
import time
import weakref

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

from zlc_data import (  # noqa: E402
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_spec_for,
)
from zlc_frontend import (  # noqa: E402
    DataFigure,
    ImageDisplayState,
    ImagePanelPayload,
)
from zlc_frontend.display_range import RelimMode  # noqa: E402
from zlc_frontend.figure import (  # noqa: E402
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    suggest_fit_view,
    suggest_view,
)
from zlc_frontend.image_display import ImageColormap  # noqa: E402
from zlc_frontend.qt_widgets import (  # noqa: E402
    FrozenRasterView,
    QtRasterBoard,
    ensure_qt_app,
)
from zlc_frontend.selector import (  # noqa: E402
    ImageColorLimitsCommit,
    RectangleGesture,
)
from zlc_workbench.data_figure.app import (  # noqa: E402
    create_data_figure_pane as open_data_figure_workbench,
)


@pytest.fixture
def application():
    return ensure_qt_app()


@pytest.fixture
def open_window(application):
    windows = []

    def open_one(figure, **options):
        window = open_data_figure_workbench(figure, **options)
        windows.append(window)
        return window

    try:
        yield open_one
    finally:
        for window in reversed(windows):
            if not window.closed:
                _close(application, window)


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _close(application, window) -> None:
    window.shutdown()
    _until(application, lambda: window.closed)
    application.processEvents()


def _image_figure(
    *,
    x_coordinates: tuple[float, ...] = (10.0, 12.0, 14.0, 16.0, 18.0),
    y_coordinates: tuple[float, ...] = (20.0, 22.0, 24.0, 26.0),
    coordinate_frame: CoordinateFrameId | None = CoordinateFrameId("u03c-camera"),
    with_fit: bool = False,
    figure_class=DataFigure,
) -> DataFigure:
    repeat = AxisSpec(AxisId("u03c.repeat"), "Repeat", REPEAT, 2, (0, 1))
    point = AxisSpec(AxisId("u03c.point"), "Point", SCAN_POINT, 1, (0,))
    y_axis = AxisSpec(
        AxisId("u03c.y"),
        "Sensor row",
        SPATIAL_Y,
        len(y_coordinates),
        y_coordinates,
        "pixel",
        coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("u03c.x"),
        "Sensor column",
        SPATIAL_X,
        len(x_coordinates),
        x_coordinates,
        "pixel",
        coordinate_frame,
    )
    yy, xx = np.meshgrid(
        np.arange(y_axis.size, dtype=np.float64),
        np.arange(x_axis.size, dtype=np.float64),
        indexing="ij",
    )
    base = 3.0 + 20.0 * np.exp(
        -((xx - 0.45 * x_axis.size) ** 2 + (yy - 0.55 * y_axis.size) ** 2)
        / 3.0
    )
    values = np.stack((base, base + 2.0), axis=0)[:, None, :, :]
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[:, :, 1, 2] = False
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            values.dtype,
            value_unit="photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("u03c-image-block"),
        DatasetRevision(4),
        values,
        ComponentValidity((y_axis.axis_id, x_axis.axis_id), valid),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03c-image-generation")),
        block,
    )
    dataset_id = DatasetId("u03c-image-dataset")
    result = None
    if with_fit:
        result = bind_fit(
            fit_spec_for(schema, "radial_gaussian_center"),
            schema,
        ).run(snapshot)
        suggestion = suggest_fit_view(schema, result)
    else:
        suggestion = suggest_view(schema, ViewIntent.IMAGE)
    assert suggestion.spec is not None
    document = FigureDocument(
        "u03c-image-document",
        2,
        (DatasetDescriptor(dataset_id, "camera", schema.fingerprint),),
        (FigureLayer("image", dataset_id, suggestion.spec),),
    )
    return figure_class(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results=None if result is None else {"image": result},
    )


def _typed_front(window):
    board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
    assert board is not None and board.front_frame is not None
    frame = board.front_frame
    payload = frame.panels[0].display_payload
    assert isinstance(payload, ImagePanelPayload)
    return board, frame, payload


def _image_target(board: QtRasterBoard):
    binding = board._image_bindings["generic-typed"]
    target = board._selector_target(binding)
    assert target is not None
    return binding, target[0]


def _point(target, x_fraction: float, y_fraction: float):
    return QtCore.QPoint(
        target.left() + int(x_fraction * max(1, target.width() - 1)),
        target.top() + int(y_fraction * max(1, target.height() - 1)),
    )


def _wheel(board: QtRasterBoard, position: QtCore.QPoint, delta: int):
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
    return event


def _drag_move(
    board: QtRasterBoard,
    position: QtCore.QPoint,
    button,
) -> None:
    board.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(position),
            QtCore.Qt.NoButton,
            button,
            QtCore.Qt.NoModifier,
        )
    )


def test_image_front_preserves_exact_axes_validity_and_all_display_interactions(
    application,
    open_window,
    tmp_path: Path,
) -> None:
    figure = _image_figure()
    exact = figure.evaluated.layers[0].cells[0].series[0].data
    window = open_window(figure)
    _until(application, lambda: window.raster_ready)

    board, frame, payload = _typed_front(window)
    assert payload.image is exact
    assert payload.value_unit == "photoelectron"
    assert payload.viewport.coordinate_frame == CoordinateFrameId("u03c-camera")
    assert payload.viewport.x_axis.axis_id == AxisId("u03c.x")
    assert payload.viewport.y_axis.axis_id == AxisId("u03c.y")
    assert not bool(payload.image.validity[1, 2])
    raster = frame.panels[0].raster
    assert raster.width > 0 and raster.height > 0
    assert len(raster.pixels) == raster.width * raster.height * 4
    assert window._display == ImageDisplayState()

    binding, target = _image_target(board)
    QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=target.center())
    assert binding.cross is not None
    assert binding.cross.x_coordinate == pytest.approx(14.0, abs=0.05)
    assert binding.cross.y_coordinate == pytest.approx(23.0, abs=0.05)
    assert not binding.cross.valid

    origin = board.visible_image_origin("generic-typed")
    assert origin is not None
    rectangle = RectangleGesture(
        origin.panel_id,
        origin.board_id,
        origin.layout_generation,
        origin.sequence,
        origin.source_identity,
        (0.2, 0.25, 0.8, 0.75),
        payload.viewport.viewport_revision,
    )
    window._accept_image_rectangle(rectangle)
    assert binding.applied_bounds == rectangle.normalized_bounds
    assert window._display == ImageDisplayState()
    assert "DISPLAY ONLY rectangle" in window._diagnostic.text()

    assert _wheel(board, target.center(), -120).isAccepted()
    _until(application, lambda: window.raster_ready and window._display.revision == 1)
    _board, _frame, zoomed = _typed_front(window)
    assert zoomed.viewport.visible_bounds != (0.0, 0.0, 1.0, 1.0)
    assert zoomed.image is exact

    binding, target = _image_target(board)
    start = target.center()
    end = QtCore.QPoint(start.x() + 12, start.y() + 7)
    QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=start)
    _drag_move(board, end, QtCore.Qt.MiddleButton)
    QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=end)
    _until(application, lambda: window.raster_ready and window._display.revision == 2)
    _board, _frame, panned = _typed_front(window)
    assert panned.viewport.visible_bounds != zoomed.viewport.visible_bounds
    assert panned.image is exact

    board.set_image_rectangle_candidate(None, panel_id="generic-typed")
    _binding, target = _image_target(board)
    QtTest.QTest.mouseDClick(board, QtCore.Qt.MiddleButton, pos=target.center())
    _until(application, lambda: window.raster_ready and window._display.revision == 3)
    _board, _frame, home = _typed_front(window)
    assert home.viewport.x_limits == home.viewport.home_x_limits
    assert home.viewport.y_limits == home.viewport.home_y_limits
    assert home.image is exact

    QtTest.QTest.mouseDClick(board, QtCore.Qt.RightButton, pos=target.center())
    assert binding.cross is None

    origin = board.visible_image_origin("generic-typed")
    assert origin is not None
    window._accept_image_interaction(
        ImageColorLimitsCommit(origin, (5.0, 19.0))
    )
    _until(application, lambda: window.raster_ready and window._display.revision == 4)
    _board, _frame, fixed = _typed_front(window)
    assert window._display.relim_mode is RelimMode.FIXED
    assert fixed.color_limits == (5.0, 19.0)
    assert fixed.image is exact

    editor = window._edit_display
    assert editor is not None
    colormap = editor._form.widget_for("colormap")
    assert isinstance(colormap, QtWidgets.QComboBox)
    index = colormap.findData(ImageColormap.INFERNO)
    assert index >= 0
    colormap.setCurrentIndex(index)
    editor._form.changed.emit("colormap")
    editor._apply_button.click()
    _until(application, lambda: window.raster_ready and window._display.revision == 5)
    _board, _frame, recolored = _typed_front(window)
    assert recolored.image is exact
    assert window._edit_display is not None
    assert window._setting_display is not None
    assert window._edit_display._base_semantic_identity == window._display
    assert window._setting_display._base_semantic_identity == window._display

    destination = tmp_path / "current-image.png"
    window._start_export(destination)
    _until(
        application,
        lambda: window._future is None and destination.exists(),
    )
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert window._status.text() == "READY"

    _close(application, window)
    assert board.front_frame is None


@pytest.mark.parametrize(
    ("figure", "reason"),
    (
        (_image_figure(coordinate_frame=None), "explicit coordinate frame"),
        (
            _image_figure(x_coordinates=(0.0, 1.0, 3.0, 4.0, 5.0)),
            "not exactly regular",
        ),
        (
            _image_figure(with_fit=True),
            "exactly one layer, cell, and input",
        ),
    ),
)
def test_image_authority_or_geometry_gaps_fail_to_whole_figure_encoded(
    application,
    open_window,
    figure: DataFigure,
    reason: str,
) -> None:
    window = open_window(figure)
    _until(application, lambda: window.raster_ready or "FAILED" in window._status.text())
    assert window.raster_ready
    assert window._view_family == "encoded"
    assert reason in window._summary.text()
    assert window.findChild(FrozenRasterView) is not None
    assert window.findChild(QtRasterBoard, "figureViewerTypedBoard").front_frame is None
    _close(application, window)


@pytest.mark.parametrize("forgery", ("values", "axis-frame"))
def test_image_rerender_cas_rejects_a_new_exact_data_object(
    application,
    open_window,
    forgery: str,
) -> None:
    from zlc_frontend.image_display import image_viewport_for_display_state
    from zlc_frontend.image_view import image_viewport_for_evaluated_image

    window = open_window(_image_figure())
    _until(application, lambda: window.raster_ready)
    board, original_frame, original_payload = _typed_front(window)
    original_renderer = window._typed_renderer

    def wrong_exact_data(*args, **kwargs):
        front = original_renderer(*args, **kwargs)
        panel = front.frame.panels[0]
        payload = panel.display_payload
        assert isinstance(payload, ImagePanelPayload)
        if forgery == "values":
            forged_image = replace(
                payload.image,
                values=payload.image.values.copy(),
            )
            forged_viewport = payload.viewport
        else:
            forged_frame = CoordinateFrameId("forged-camera-frame")
            forged_image = replace(
                payload.image,
                x_axis=replace(
                    payload.image.x_axis,
                    coordinate_frame=forged_frame,
                ),
                y_axis=replace(
                    payload.image.y_axis,
                    coordinate_frame=forged_frame,
                ),
            )
            forged_viewport = image_viewport_for_display_state(
                front.state,
                image_viewport_for_evaluated_image(forged_image),
            )
        forged_payload = replace(
            payload,
            image=forged_image,
            viewport=forged_viewport,
        )
        forged_frame = replace(
            front.frame,
            panels=(replace(panel, display_payload=forged_payload),),
        )
        return replace(front, frame=forged_frame)

    window._typed_renderer = wrong_exact_data
    window._start_typed_render(
        replace(window._display, revision=1, colormap=ImageColormap.VIRIDIS)
    )
    _until(application, lambda: window._future is None)
    expected = (
        "changed frozen evaluated data"
        if forgery == "values"
        else "changed frozen source provenance"
    )
    assert expected in window._diagnostic.text()
    assert board.front_frame is original_frame
    assert board.front_frame.panels[0].display_payload is original_payload
    assert window._display == ImageDisplayState()
    assert window.raster_ready
    _close(application, window)


def test_initial_image_front_cannot_self_attest_a_conflicting_viewport() -> None:
    import zlc_workbench.data_figure.render_lane as figure_module
    from zlc_workbench.data_figure.window import DataFigureWindow
    from zlc_frontend.image_view import ImageViewportTransform

    figure = _image_figure()
    front = figure_module._render_typed_front(
        figure,
        ImageDisplayState(),
        current_value_limits=None,
        previous_relim_mode=None,
        previous_count_scale=None,
        sequence=0,
        cancelled=threading.Event(),
    )
    panel = front.frame.panels[0]
    payload = panel.display_payload
    assert isinstance(payload, ImagePanelPayload)
    forged_viewport = ImageViewportTransform(
        payload.viewport.axes,
        0,
        (0.0, 0.25, 1.0, 0.75),
    )
    forged_payload = replace(payload, viewport=forged_viewport)
    forged = replace(
        front,
        frame=replace(
            front.frame,
            panels=(replace(panel, display_payload=forged_payload),),
        ),
    )
    with pytest.raises(ValueError, match="conflicting authored state"):
        DataFigureWindow._validate_authored_front(
            forged,
            ImageDisplayState(),
        )


def test_image_control_construction_failure_keeps_exact_front_fail_closed(
    application,
    open_window,
    monkeypatch,
) -> None:
    import zlc_workbench.data_figure.window as figure_module

    figure = _image_figure()
    exact = figure.evaluated.layers[0].cells[0].series[0].data

    def fail_controls(*_args, **_kwargs):
        raise RuntimeError("forced image controls failure")

    monkeypatch.setattr(
        figure_module,
        "FluentRevisionedFormEditor",
        fail_controls,
    )
    window = open_window(figure)
    try:
        _until(application, lambda: window.raster_ready)
        board, _frame, payload = _typed_front(window)
        assert payload.image is exact
        assert window._status.text() == "TYPED CONTROLS FAILED"
        assert "forced image controls failure" in window._diagnostic.text()
        assert board._image_bindings == {}
        assert not window._settings_button.isEnabled()
        assert not window._interaction_switch.isEnabled()
    finally:
        if not window.closed:
            _close(application, window)


class _WeakDataFigure(DataFigure):
    __slots__ = ("__weakref__",)


def test_close_during_blocked_image_rerender_has_no_late_present_and_releases_cache(
    application,
    open_window,
) -> None:
    figure = _image_figure(figure_class=_WeakDataFigure)
    reference = weakref.ref(figure)
    window = open_window(figure)
    entered = threading.Event()
    release = threading.Event()
    try:
        _until(application, lambda: window.raster_ready)
        board, admitted, _payload = _typed_front(window)
        original_renderer = window._typed_renderer

        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(10.0):
                raise TimeoutError("blocked IMAGE rerender was not released")
            return original_renderer(*args, **kwargs)

        window._typed_renderer = blocked
        window._start_typed_render(
            replace(
                window._display,
                revision=1,
                colormap=ImageColormap.VIRIDIS,
            )
        )
        assert entered.wait(5.0)
        window.shutdown()
        application.processEvents()
        assert not window.closed
        assert board.front_frame is None
        release.set()
        _until(application, lambda: window.closed)
        application.processEvents()
        assert board.front_frame is None
        assert admitted.sequence == 0
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
    del figure
    del blocked
    del original_renderer
    del window
    gc.collect()
    assert reference() is None
