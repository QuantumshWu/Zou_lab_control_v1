"""Narrow evidence for the formal PlotPanel IMAGE hot path.

These tests deliberately stop the worker answer while real Qt pointer events
continue.  The widget must author one exact request plus one latest desired
state; it must never compose Matplotlib on the GUI thread, lose accumulated
wheel/pan input, promote the immutable camera dtype, or rebase an Area
gesture when the delayed answer arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import statistics
import time

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@dataclass
class _ImageHarness:
    application: object
    value: object
    composer: object
    display: object
    host: object
    logical_size: tuple[int, int]

    def close(self) -> None:
        self.host.close()
        self.composer.close()
        self.application.processEvents()


def _painted_image_surface(harness: _ImageHarness):
    """Return the exact public front payload and its Qt image hit box."""

    from gui_user_flow import raster_subrect

    frame = harness.host.front_frame
    assert frame is not None and len(frame.panels) == 1
    payload = frame.panels[0].display_payload
    target = raster_subrect(
        harness.host.board.rect(),
        payload.raster_geometry.image_bounds,
    )
    return payload, target


def _image_value(*, revision: int, side: int, dtype) -> object:
    """Build one current immutable camera signal without Console test seams."""

    import zlc_data as data
    from zlc_neutral_atom.processing.signal_plane import SignalValue

    side = int(side)
    if side < 2:
        raise ValueError("image side must be at least two")
    repeat = data.AxisSpec(data.AxisId("profile.repeat"), "repeat", data.REPEAT, 1, (0,))
    coordinate_frame = data.CoordinateFrameId("profile.camera")
    y_axis = data.AxisSpec(
        data.AxisId("profile.y"),
        "camera y",
        data.SPATIAL_Y,
        side,
        tuple(range(side)),
        "pixel",
        coordinate_frame,
    )
    x_axis = data.AxisSpec(
        data.AxisId("profile.x"),
        "camera x",
        data.SPATIAL_X,
        side,
        tuple(range(side)),
        "pixel",
        coordinate_frame,
    )
    values = (
        np.arange(side * side, dtype=np.uint32).reshape(side, side) % 251
    ).astype(dtype, copy=False)[None, None, :, :]
    value_schema = data.ValueSchema(
        (y_axis, x_axis),
        data.ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        values.dtype,
        value_unit="count",
    )
    schema = data.DatasetSchema(repeat, data.PointTable(1), None, value_schema)
    block = data.DataBlock(
        data.BlockId("profile-image"),
        data.DatasetRevision(int(revision)),
        values,
        data.DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    snapshot = data.OwnedSnapshot(
        block.ref(data.StreamGenerationId("profile-image-generation")),
        block,
    )
    return SignalValue(
        name="image",
        snapshot=snapshot,
        coverage=None,
    )


def _harness(*, side: int = 512) -> _ImageHarness:
    from zlc_frontend import ImageDisplayState
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.panel_render import PanelComposer
    from zlc_frontend.plot_layout import panel_surface_geometry
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    value = _image_value(revision=1, side=side, dtype=np.uint8)
    view = suggest_view(value.schema, ViewIntent.IMAGE).spec
    assert view is not None
    composer = PanelComposer(
        "profile-image",
        intent=ViewIntent.IMAGE,
        view=view,
    )
    display = ImageDisplayState()
    frame, _figure = composer.compose_with_figure(
        value.snapshot,
        display=display,
    )
    logical_size = panel_surface_geometry("2x2").logical_size
    host = SinglePanelHost("profile-image", group="profile-image")
    host.present_frame(frame, logical_size=logical_size)
    host.show()
    application.processEvents()
    host.set_selectors_enabled(True)
    return _ImageHarness(
        application,
        value,
        composer,
        display,
        host,
        logical_size,
    )


def _answer_viewport_commands(harness, commands, *, start: int = 0) -> int:
    from zlc_frontend.image_display import image_display_for_viewport
    from zlc_frontend.selector import ImageViewportCommit

    index = start
    answered = 0
    while index < len(commands):
        command = commands[index]
        index += 1
        if not isinstance(command, ImageViewportCommit):
            continue
        harness.display = image_display_for_viewport(
            harness.display,
            command.viewport,
        )
        frame, _figure = harness.composer.compose_with_figure(
            harness.value.snapshot,
            display=harness.display,
        )
        harness.host.present_frame(frame, logical_size=harness.logical_size)
        harness.application.processEvents()
        answered += 1
    return answered


def test_wheel_and_pan_are_exact_latest_only_without_gui_agg(monkeypatch) -> None:
    from PyQt5 import QtCore, QtTest
    from gui_user_flow import drag_mouse_move, point_in_rect, send_wheel

    harness = _harness()
    try:
        board = harness.host.board
        payload, target = _painted_image_surface(harness)
        centre = point_in_rect(target, 0.5, 0.5)
        commands: list[object] = []
        harness.host.viewCommitted.connect(commands.append)

        renderer = harness.composer._renderer
        canvas_type = type(renderer._figure.canvas)
        ordinary_draw = canvas_type.draw
        full_draws = 0

        def counted_draw(canvas, *args, **kwargs):
            nonlocal full_draws
            full_draws += 1
            return ordinary_draw(canvas, *args, **kwargs)

        monkeypatch.setattr(canvas_type, "draw", counted_draw)

        expected = payload.viewport
        wheel_latency_ms = []
        for _step in range(25):
            expected = expected.centered_zoom((0.5, 0.5), 1.0 / 1.1)
            started = time.perf_counter_ns()
            send_wheel(board, centre, -120)
            wheel_latency_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )

        # One worker answer is in flight; every later event accumulated into
        # the single exact latest state rather than being dropped or queued.
        assert len(commands) == 1
        assert statistics.median(wheel_latency_ms) < 10.0
        assert _answer_viewport_commands(harness, commands) == 2
        assert len(commands) == 2
        payload, _target = _painted_image_surface(harness)
        assert payload.viewport.visible_bounds == expected.visible_bounds
        # Stable endpoint chrome means the first answer primes the one Axes
        # background and the second exact answer needs no complete Figure draw.
        assert full_draws == 1

        assert harness.value.dtype == np.dtype(np.uint8)
        assert payload.image.values.dtype == np.dtype(np.uint8)

        # Pan is absolute from the press-time transform.  Thirty move events
        # must therefore end at the exact final pointer displacement, again as
        # one in-flight answer plus one latest answer.
        payload, target = _painted_image_surface(harness)
        centre = point_in_rect(target, 0.5, 0.5)
        origin = payload.viewport
        final_delta = (30.0, 15.0)
        expected = origin.panned_by_pixels(
            final_delta,
            (max(1, target.width()), max(1, target.height())),
        )
        start = len(commands)
        pan_latency_ms = []
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.MiddleButton,
            pos=centre,
        )
        for step in range(1, 31):
            point = QtCore.QPoint(
                centre.x() + step,
                centre.y() + step // 2,
            )
            started = time.perf_counter_ns()
            drag_mouse_move(board, point, QtCore.Qt.MiddleButton)
            pan_latency_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=QtCore.QPoint(
                centre.x() + int(final_delta[0]),
                centre.y() + int(final_delta[1]),
            ),
        )
        assert len(commands) == start + 1
        assert statistics.median(pan_latency_ms) < 10.0
        assert _answer_viewport_commands(harness, commands, start=start) == 2
        assert len(commands) == start + 2
        payload, _target = _painted_image_surface(harness)
        assert payload.viewport.visible_bounds == expected.visible_bounds
    finally:
        harness.close()


def test_pending_render_keeps_cross_and_area_on_exact_painted_front() -> None:
    from PyQt5 import QtCore, QtTest
    from gui_user_flow import drag_mouse_move, point_in_rect, send_wheel
    from zlc_frontend.image_display import image_display_for_viewport

    harness = _harness()
    try:
        board = harness.host.board
        payload, target = _painted_image_surface(harness)
        centre = point_in_rect(target, 0.5, 0.5)
        old_viewport = payload.viewport
        old_origin = harness.host.visible_interaction_origin()
        assert old_origin is not None
        commands: list[object] = []
        crosses: list[object] = []
        areas: list[object] = []
        selections: list[object] = []
        harness.host.viewCommitted.connect(commands.append)
        harness.host.crossSelected.connect(crosses.append)

        def capture_area(gesture) -> None:
            areas.append(gesture)
            selections.append(
                harness.host.selection_for_rectangle_gesture(gesture)
            )

        harness.host.rectangleSelected.connect(capture_area)

        # Hold the first wheel answer.  Cross is an immediate sample of the
        # still-painted immutable payload and must not vanish behind that wait.
        send_wheel(board, centre, -120)
        assert len(commands) == 1
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=centre,
        )
        assert len(crosses) == 1
        assert crosses[0].origin.painted_revision == old_viewport.viewport_revision

        # Begin Area on that same exact front, then admit the delayed viewport
        # answer before release.  The board may retain the new front, but the
        # active held raster and coordinate mapping must remain the old pair.
        start = point_in_rect(target, 0.25, 0.25)
        finish = point_in_rect(target, 0.70, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, finish, QtCore.Qt.LeftButton)
        assert harness.host.visible_interaction_origin() == old_origin

        command = commands[0]
        harness.display = image_display_for_viewport(
            harness.display,
            command.viewport,
        )
        answer, _figure = harness.composer.compose_with_figure(
            harness.value.snapshot,
            display=harness.display,
        )
        harness.host.present_frame(answer, logical_size=harness.logical_size)
        harness.application.processEvents()
        payload = harness.host.front_frame.panels[0].display_payload
        assert payload.viewport == command.viewport
        assert harness.host.visible_interaction_origin() == old_origin

        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=finish)
        assert len(areas) == len(selections) == 1
        assert areas[0].viewport_revision == old_viewport.viewport_revision
        assert selections[0] is not None
        assert harness.host.visible_interaction_origin() != old_origin
        payload = harness.host.front_frame.panels[0].display_payload
        assert payload.viewport == command.viewport
    finally:
        harness.close()
