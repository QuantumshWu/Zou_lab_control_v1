from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _input(dataset_name: str, sequence: int):
    from zlc_data import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
    from zlc_frontend.figure import DatasetId, EvaluatedInput

    schema = ("a" if dataset_name == "curve" else "b") * 64
    dataset_id = DatasetId(dataset_name)
    ref = DatasetRevisionRef(
        BlockId(f"{dataset_name}-block"),
        StreamGenerationId(f"{dataset_name}-generation"),
        schema,
        DatasetRevision(sequence + 1),
    )
    return EvaluatedInput(dataset_id, ref)


def _curve_panel(sequence: int, *, display_revision: int = 0, offset: float = 0.0):
    from zlc_data import AxisId, MONITOR_HISTORY
    from zlc_frontend.curve_display import CurveViewportTransform
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedCurve, EvaluatedSeries
    from zlc_frontend.render import (
        CoherenceStamp,
        CurvePanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        PixelFormat,
        RasterBuffer,
        SourceIdentity,
    )

    evaluated_input = _input("curve", sequence)
    axis = EvaluatedAxis(
        AxisId("monitor.history"),
        "Shots ago",
        MONITOR_HISTORY,
        "ms",
        (0, 1, 2, 3),
        (0.0, 1.0, 2.0, 3.0),
    )
    first = EvaluatedCurve(
        axis,
        "count",
        np.asarray((0.0, 1.0, 2.0, 3.0)) + offset,
        np.asarray((True, True, False, True)),
    )
    second = EvaluatedCurve(
        axis,
        "count",
        np.asarray((3.0, 2.0, 1.0, 0.0)) + offset,
        np.asarray((True, True, True, True)),
    )
    viewport = CurveViewportTransform(
        axis,
        display_revision,
        (0.20, 0.10, 0.80, 0.90),
        (-0.5, 3.5),
        (-1.0 + offset, 4.0 + offset),
        (-0.15, 3.15),
    )
    payload = CurvePanelPayload(
        evaluated_input,
        viewport,
        (EvaluatedSeries((), first), EvaluatedSeries((), second)),
        ("site 0", "site 1"),
    )
    presentation = PanelPresentationIdentity(
        "curve", "curve-document", 0, 0, display_revision
    )
    stamp = CoherenceStamp(
        "run",
        f"curve-epoch-{sequence}",
        "curve-frame",
        evaluated_input.ref.schema_fingerprint,
        "c" * 64,
        (evaluated_input,),
        (presentation,),
    )
    source = SourceIdentity(
        evaluated_input.dataset_id,
        evaluated_input.ref.block_id,
        evaluated_input.ref.stream_generation,
        evaluated_input.ref.schema_fingerprint,
    )
    raster = RasterBuffer(
        200,
        100,
        800,
        PixelFormat.RGBA8888,
        bytes((20, 30, 40, 255)) * (200 * 100),
    )
    return PanelFrame("curve", "curve", source, stamp, raster, payload)


def _image_panel(sequence: int, *, viewport_revision: int = 0):
    from zlc_data import (
        AxisId,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedImage
    from zlc_frontend.image_view import ImageViewportTransform
    from zlc_frontend.render import (
        CoherenceStamp,
        ImagePanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        PixelFormat,
        RasterBuffer,
        SourceIdentity,
    )

    evaluated_input = _input("image", sequence)
    frame = CoordinateFrameId("camera")
    y_spec = AxisSpec(
        AxisId("camera.y"),
        "y",
        SPATIAL_Y,
        2,
        (0.0, 1.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    x_spec = AxisSpec(
        AxisId("camera.x"),
        "x",
        SPATIAL_X,
        2,
        (0.0, 1.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    viewport = ImageViewportTransform(
        (y_spec, x_spec), viewport_revision=viewport_revision
    )
    x_axis = EvaluatedAxis(x_spec.axis_id, "x", SPATIAL_X, "pixel", (0, 1), (0.0, 1.0))
    y_axis = EvaluatedAxis(y_spec.axis_id, "y", SPATIAL_Y, "pixel", (0, 1), (0.0, 1.0))
    image = EvaluatedImage(
        x_axis,
        y_axis,
        np.asarray(((1.0, 2.0), (3.0, 4.0 + sequence))),
        np.ones((2, 2), dtype=bool),
    )
    histogram = (1, 1, 1, 1) + (0,) * 251
    payload = ImagePanelPayload(
        image,
        evaluated_input,
        viewport,
        (1.0, 4.0 + sequence),
        histogram,
        tuple(0xFF000000 | index for index in range(256)),
        (0.0, 10.0),
    )
    presentation = PanelPresentationIdentity(
        "image", "image-document", 0, 0, viewport_revision
    )
    stamp = CoherenceStamp(
        "run",
        f"image-epoch-{sequence}",
        "image-frame",
        evaluated_input.ref.schema_fingerprint,
        "d" * 64,
        (evaluated_input,),
        (presentation,),
    )
    source = SourceIdentity(
        evaluated_input.dataset_id,
        evaluated_input.ref.block_id,
        evaluated_input.ref.stream_generation,
        evaluated_input.ref.schema_fingerprint,
    )
    raster = RasterBuffer(2, 2, 2, PixelFormat.INDEXED8, bytes((1, 2, 3, 4)))
    return PanelFrame("image", "image", source, stamp, raster, payload)


def _frame(sequence: int, *, curve_revision: int = 0, offset: float = 0.0):
    from zlc_frontend.render import BoardFrame

    return BoardFrame(
        "curve-board",
        0,
        sequence,
        (
            _image_panel(sequence),
            _curve_panel(sequence, display_revision=curve_revision, offset=offset),
        ),
    )


def _board(frame, curve_commands, image_commands):
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    board = QtRasterBoard(("image", "curve"), columns=2)
    board.resize(800, 300)
    board.show()
    board.present(frame)
    image_payload = frame.panels[0].display_payload
    board.bind_rectangle_selector(
        "image",
        image_payload.viewport,
        lambda _gesture: None,
        interaction_callback=image_commands.append,
    )
    board.bind_curve_interaction("curve", curve_commands.append)
    application.processEvents()
    return application, board


def _curve_target(board):
    binding = board._numeric_binding_for_kind("curve")
    assert binding is not None
    target = board._numeric_target(binding)
    assert target is not None
    return target


def _point(rect, x_fraction: float, y_fraction: float):
    from PyQt5 import QtCore

    return QtCore.QPoint(
        int(round(rect.left() + x_fraction * rect.width())),
        int(round(rect.top() + y_fraction * rect.height())),
    )


def _wheel(board, position, delta: int):
    from PyQt5 import QtCore, QtGui

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


def _drag_move(board, position, button) -> None:
    from PyQt5 import QtCore, QtGui

    board.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(position),
            QtCore.Qt.NoButton,
            button,
            QtCore.Qt.NoModifier,
        )
    )


def _double_click(board, position, button) -> None:
    from PyQt5 import QtCore, QtGui

    board.mouseDoubleClickEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonDblClick,
            QtCore.QPointF(position),
            button,
            button,
            QtCore.Qt.NoModifier,
        )
    )


def _accepted_curve_frame(sequence: int, command):
    accepted = _frame(sequence, curve_revision=command.viewport.display_revision)
    payload = accepted.panels[1].display_payload
    payload = replace(
        payload,
        viewport=replace(payload.viewport, x_limits=command.viewport.x_limits),
    )
    return replace(
        accepted,
        panels=(
            accepted.panels[0],
            replace(accepted.panels[1], display_payload=payload),
        ),
    )


def test_curve_uses_draw_frozen_bbox_and_horizontal_span() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture

    commands: list[object] = []
    application, board = _board(_frame(1), commands, [])
    try:
        plot = _curve_target(board).plot
        assert plot.left() == 480.0
        assert plot.top() == 30.0
        assert plot.width() == pytest.approx(240.0)
        assert plot.height() == pytest.approx(240.0)

        start = _point(plot, 0.25, 0.50)
        end = _point(plot, 0.75, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, end, QtCore.Qt.LeftButton)
        assert board._selector_hold is not None
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert isinstance(commands[-1], CurveRangeGesture)
        assert commands[-1].x_span == pytest.approx((0.5, 2.5))
        board.set_curve_range_candidate(commands[-1].x_span)
        assert board._numeric_bindings["curve"].applied_span == pytest.approx(
            (0.5, 2.5)
        )
        assert board._selector_hold is None
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=start)
        clear = commands[-1]
        assert isinstance(clear, CurveRangeGesture) and clear.x_span is None
        board.set_curve_range_candidate(clear.x_span)
        assert board._numeric_bindings["curve"].applied_span is None
    finally:
        board.close()
        application.processEvents()


def test_curve_hover_all_series_and_continuous_cross() -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    application, board = _board(_frame(1), [], [])
    try:
        plot = _curve_target(board).plot
        # data (2, 1) is invalid in site 0 but exact and valid in site 1.
        position = _point(plot, 0.625, 0.40)
        QtTest.QTest.mouseMove(board, position)
        binding = board._numeric_bindings["curve"]
        assert binding.hover is not None
        assert binding.hover.series_label == "site 1"
        assert (binding.hover.x, binding.hover.y) == (2.0, 1.0)

        arbitrary = _point(plot, 0.33, 0.61)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=arbitrary)
        cross = binding.cross
        assert cross is not None
        assert cross.x != round(cross.x)
        board.mouseDoubleClickEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QPointF(arbitrary),
                QtCore.Qt.RightButton,
                QtCore.Qt.RightButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert binding.cross is None
    finally:
        board.close()
        application.processEvents()


def test_curve_cross_and_hover_overlay_show_both_axis_units(monkeypatch) -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    curve_commands = []
    application, board = _board(_frame(0), curve_commands, [])
    try:
        plot = _curve_target(board).plot
        position = _point(plot, 0.55, 0.45)
        QtTest.QTest.mouseMove(board, position)
        binding = board._numeric_bindings["curve"]
        assert binding.hover is not None

        labels = []

        def capture_label(_painter, label, _plot, _color, **_kwargs):
            labels.append(label)

        monkeypatch.setattr(
            type(board),
            "_paint_curve_label",
            staticmethod(capture_label),
        )
        image = QtGui.QImage(
            board.size(),
            QtGui.QImage.Format_ARGB32_Premultiplied,
        )
        painter = QtGui.QPainter(image)
        try:
            board._paint_numeric_binding_overlay(painter, binding)
        finally:
            painter.end()
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=position)
        assert binding.cross is not None
        painter = QtGui.QPainter(image)
        try:
            board._paint_numeric_binding_overlay(painter, binding)
        finally:
            painter.end()
        assert len(labels) == 2
        assert all("x=" in label and " ms" in label for label in labels)
        assert all("y=" in label and " count" in label for label in labels)
    finally:
        board.close()
        application.processEvents()


def test_curve_x_only_wheel_pan_and_area_home() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture, CurveViewportCommit

    commands: list[object] = []
    application, board = _board(_frame(1), commands, [])
    try:
        target = _curve_target(board)
        plot = target.plot
        center = _point(plot, 0.5, 0.5)
        initial_y = target.payload.viewport.y_limits
        assert _wheel(board, center, -120).isAccepted()
        zoom = commands[-1]
        assert isinstance(zoom, CurveViewportCommit)
        assert zoom.viewport.y_limits == initial_y
        assert zoom.viewport.x_limits == pytest.approx((-0.3181818182, 3.3181818182))

        # Paint the exact accepted revision before issuing another command.
        board.present(_accepted_curve_frame(2, zoom))
        target = _curve_target(board)
        plot = target.plot
        press = _point(plot, 0.50, 0.50)
        move = _point(plot, 0.60, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=press)
        _drag_move(board, move, QtCore.Qt.MiddleButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=move)
        pan = commands[-1]
        assert isinstance(pan, CurveViewportCommit)
        assert pan.viewport.y_limits == target.payload.viewport.y_limits
        assert pan.viewport.x_limits[0] < zoom.viewport.x_limits[0]

        board.present(_accepted_curve_frame(3, pan))
        target = _curve_target(board)
        plot = target.plot
        _double_click(board, _point(plot, 0.5, 0.5), QtCore.Qt.MiddleButton)
        home = commands[-1]
        assert isinstance(home, CurveViewportCommit)
        assert home.viewport.x_limits == target.payload.viewport.home_x_limits
        assert home.viewport.y_limits == target.payload.viewport.y_limits

        board.present(_accepted_curve_frame(4, home))
        plot = _curve_target(board).plot
        start = _point(plot, 0.25, 0.5)
        end = _point(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        range_gesture = commands[-1]
        assert isinstance(range_gesture, CurveRangeGesture)
        board.set_curve_range_candidate(range_gesture.x_span)
        span = board._numeric_bindings["curve"].applied_span
        assert span is not None
        _double_click(board, _point(plot, 0.5, 0.5), QtCore.Qt.MiddleButton)
        area = commands[-1]
        assert isinstance(area, CurveViewportCommit)
        assert area.viewport.x_limits == span
        assert area.viewport.y_limits == _curve_target(board).payload.viewport.y_limits
    finally:
        board.close()
        application.processEvents()


def test_curve_pending_is_panel_local_and_exact_discard() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import ImageViewportCommit

    curve_commands: list[object] = []
    image_commands: list[object] = []
    application, board = _board(_frame(1), curve_commands, image_commands)
    try:
        curve_plot = _curve_target(board).plot
        _wheel(board, _point(curve_plot, 0.5, 0.5), -120)
        curve_origin = curve_commands[-1].origin
        assert board._numeric_bindings["curve"].pending_viewport is not None

        image_target = board._selector_target()[0]
        assert _wheel(board, _point(image_target, 0.5, 0.5), -120).isAccepted()
        assert isinstance(image_commands[-1], ImageViewportCommit)
        assert board.discard_pending_curve_interaction(curve_origin)
        assert not board.discard_pending_curve_interaction(curve_origin)
    finally:
        board.close()
        application.processEvents()


def test_panel_readiness_parks_stale_curve_without_faulting_healthy_image() -> None:
    from zlc_frontend.selector import CurveViewportCommit, ImageViewportCommit

    curve_commands: list[object] = []
    image_commands: list[object] = []
    application, board = _board(_frame(1), curve_commands, image_commands)
    try:
        board.set_interaction_readiness(image=True, curve=False)
        curve_plot = _curve_target(board).plot
        stale_event = _wheel(board, _point(curve_plot, 0.5, 0.5), -120)
        assert not stale_event.isAccepted()
        assert curve_commands == []
        assert board.curve_selector_fault is None
        assert board._numeric_bindings["curve"].binding_enabled

        image_target = board._selector_target()[0]
        assert _wheel(board, _point(image_target, 0.5, 0.5), -120).isAccepted()
        assert isinstance(image_commands[-1], ImageViewportCommit)
        assert board.discard_pending_image_interaction(image_commands[-1].origin)

        board.set_interaction_readiness(image=True, curve=True)
        assert _wheel(board, _point(curve_plot, 0.5, 0.5), -120).isAccepted()
        assert isinstance(curve_commands[-1], CurveViewportCommit)
        assert board.curve_selector_fault is None
    finally:
        board.close()
        application.processEvents()


def test_curve_hold_freezes_target_while_sibling_and_front_advance() -> None:
    from PyQt5 import QtCore, QtTest

    application, board = _board(_frame(1), [], [])
    try:
        plot = _curve_target(board).plot
        start = _point(plot, 0.25, 0.5)
        end = _point(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        held = board.visible_curve_payload()
        held_pixels = board._selector_hold.prepared[0]
        board.present(_frame(2, offset=10.0))
        assert board.front_frame.sequence == 2
        assert board.visible_curve_payload() is held
        assert board._selector_hold.prepared[0] is held_pixels
        assert board.front_frame.panels[0].display_payload.image.values[1, 1] == 6.0
        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert board.visible_curve_payload() is board.front_frame.panels[1].display_payload
        assert board.visible_curve_payload() is not held
    finally:
        board.close()
        application.processEvents()


def test_extreme_curve_transform_is_rejected_inside_qt_event_boundary() -> None:
    from PyQt5 import QtCore, QtTest

    frame = _frame(1)
    curve_panel = frame.panels[1]
    payload = replace(
        curve_panel.display_payload,
        viewport=replace(
            curve_panel.display_payload.viewport,
            x_limits=(-8.0e307, 9.0e307),
        ),
    )
    frame = replace(
        frame,
        panels=(frame.panels[0], replace(curve_panel, display_payload=payload)),
    )
    commands: list[object] = []
    application, board = _board(frame, commands, [])
    try:
        plot = _curve_target(board).plot
        assert _wheel(board, _point(plot, 0.5, 0.5), 120).isAccepted()
        assert commands == []
        assert board.curve_selector_fault is None

        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.MiddleButton,
            pos=_point(plot, 0.25, 0.5),
        )
        overflow_point = _point(plot, 2.0, 0.5)
        _drag_move(board, overflow_point, QtCore.Qt.MiddleButton)
        assert board._numeric_bindings["curve"].pan_candidate is None
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=overflow_point,
        )
        assert board._selector_hold is None
        assert commands == []
        assert board.curve_selector_fault is None
    finally:
        board.close()
        application.processEvents()


def test_curve_lifecycle_and_callback_fault_are_local() -> None:
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

    application, board = _board(_frame(1), [], [])
    try:
        plot = _curve_target(board).plot
        start = _point(plot, 0.2, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.keyClick(board, QtCore.Qt.Key_Escape)
        assert board._selector_hold is None
        assert board._numeric_bindings["curve"].span_candidate is None

        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        assert board._selector_hold is not None
        QtWidgets.QApplication.sendEvent(board, QtGui.QHideEvent())
        assert board._selector_hold is None
        assert board._numeric_bindings["curve"].span_candidate is None

        board.show()
        application.processEvents()
        plot = _curve_target(board).plot
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(plot, 0.2, 0.5),
        )
        board.present(_frame(2, curve_revision=1))
        assert board._selector_hold is None

        board.bind_curve_interaction("curve", lambda _command: None)
        plot = _curve_target(board).plot
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(plot, 0.2, 0.5),
        )
        board.unbind_curve_interaction()
        assert board._selector_hold is None

        with pytest.raises(ValueError, match="board-wide enabled state"):
            board.bind_curve_interaction(
                "curve",
                lambda _command: None,
                enabled=False,
            )

        board.bind_curve_interaction(
            "curve",
            lambda _command: (_ for _ in ()).throw(RuntimeError("curve boom")),
        )
        assert _wheel(
            board,
            _point(_curve_target(board).plot, 0.5, 0.5),
            -120,
        ).isAccepted()
        assert board.curve_selector_fault is not None
        assert not board._numeric_bindings["curve"].binding_enabled
        assert board._image_binding_enabled
    finally:
        board.close()
        application.processEvents()
