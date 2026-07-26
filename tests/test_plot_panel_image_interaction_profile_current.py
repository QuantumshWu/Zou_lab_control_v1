"""Narrow evidence for the formal PlotPanel IMAGE hot path.

These tests deliberately stop the worker answer while real Qt pointer events
continue.  The widget must author one exact request plus one latest desired
state; it must never compose Matplotlib on the GUI thread, lose accumulated
wheel/pan input, copy/promote the immutable camera plane, or rebase an Area
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
    provenance: object
    display: object
    host: object
    logical_size: tuple[int, int]

    def close(self) -> None:
        self.host.close()
        self.composer.close()
        self.application.processEvents()


def _harness(*, side: int = 512) -> _ImageHarness:
    from test_task_console_figure_current import _image_value
    from zlc_frontend import ImageDisplayState
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.panel_render import PanelComposer, PanelProvenance
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
    provenance = PanelProvenance("profile-run", "profile-epoch", "0" * 64)
    display = ImageDisplayState()
    frame, _figure = composer.compose_with_figure(
        value.snapshot,
        display=display,
        provenance=provenance,
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
        provenance,
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
            provenance=harness.provenance,
        )
        harness.host.present_frame(frame, logical_size=harness.logical_size)
        harness.application.processEvents()
        answered += 1
    return answered


def test_wheel_and_pan_are_exact_latest_only_without_gui_agg(monkeypatch) -> None:
    from PyQt5 import QtCore, QtTest
    from test_u02c_qt_curve_interaction import _drag_move, _point, _wheel

    harness = _harness()
    try:
        board = harness.host.board
        binding = board._image_bindings["profile-image"]
        target = board._selector_target(binding)[0]
        centre = _point(target, 0.5, 0.5)
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

        expected = binding.viewport
        wheel_latency_ms = []
        for _step in range(25):
            expected = expected.centered_zoom((0.5, 0.5), 1.0 / 1.1)
            started = time.perf_counter_ns()
            _wheel(board, centre, -120)
            wheel_latency_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )

        # One worker answer is in flight; every later event accumulated into
        # the single exact latest state rather than being dropped or queued.
        assert len(commands) == 1
        assert binding.queued_viewport_bounds == expected.visible_bounds
        assert statistics.median(wheel_latency_ms) < 10.0
        assert _answer_viewport_commands(harness, commands) == 2
        assert len(commands) == 2
        assert (
            board.visible_image_payload().viewport.visible_bounds
            == expected.visible_bounds
        )
        # Stable endpoint chrome means the first answer primes the one Axes
        # background and the second exact answer needs no complete Figure draw.
        assert full_draws == 1

        prepared = renderer._prepared_image_value
        assert prepared is not None
        assert prepared[0].dtype == np.dtype(np.uint8)
        assert np.shares_memory(
            prepared[0],
            harness.value.snapshot.block.values,
        )

        # Pan is absolute from the press-time transform.  Thirty move events
        # must therefore end at the exact final pointer displacement, again as
        # one in-flight answer plus one latest answer.
        binding = board._image_bindings["profile-image"]
        target = board._selector_target(binding)[0]
        centre = _point(target, 0.5, 0.5)
        origin = binding.viewport
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
            _drag_move(board, point, QtCore.Qt.MiddleButton)
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
        assert binding.queued_viewport_bounds == expected.visible_bounds
        assert statistics.median(pan_latency_ms) < 10.0
        assert _answer_viewport_commands(harness, commands, start=start) == 2
        assert len(commands) == start + 2
        assert (
            board.visible_image_payload().viewport.visible_bounds
            == expected.visible_bounds
        )
    finally:
        harness.close()


def test_pending_render_keeps_cross_and_area_on_exact_painted_front() -> None:
    from PyQt5 import QtCore, QtTest
    from test_u02c_qt_curve_interaction import _drag_move, _point, _wheel
    from zlc_frontend.image_display import image_display_for_viewport

    harness = _harness()
    try:
        board = harness.host.board
        binding = board._image_bindings["profile-image"]
        target = board._selector_target(binding)[0]
        centre = _point(target, 0.5, 0.5)
        old_viewport = board.visible_image_payload().viewport
        commands: list[object] = []
        crosses: list[object] = []
        areas: list[object] = []
        selections: list[object] = []
        harness.host.viewCommitted.connect(commands.append)
        harness.host.crossSelected.connect(crosses.append)

        def capture_area(gesture) -> None:
            areas.append(gesture)
            selections.append(board.selection_for_rectangle_gesture(gesture))

        harness.host.rectangleSelected.connect(capture_area)

        # Hold the first wheel answer.  Cross is an immediate sample of the
        # still-painted immutable payload and must not vanish behind that wait.
        _wheel(board, centre, -120)
        assert len(commands) == 1
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=centre,
        )
        assert len(crosses) == 1
        assert (
            crosses[0].origin.presentation.panel_revision
            == old_viewport.viewport_revision
        )

        # Begin Area on that same exact front, then admit the delayed viewport
        # answer before release.  The board may retain the new front, but the
        # active held raster and coordinate mapping must remain the old pair.
        start = _point(target, 0.25, 0.25)
        finish = _point(target, 0.70, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, finish, QtCore.Qt.LeftButton)
        assert board._selector_hold is not None

        command = commands[0]
        harness.display = image_display_for_viewport(
            harness.display,
            command.viewport,
        )
        answer, _figure = harness.composer.compose_with_figure(
            harness.value.snapshot,
            display=harness.display,
            provenance=harness.provenance,
        )
        harness.host.present_frame(answer, logical_size=harness.logical_size)
        harness.application.processEvents()
        assert board.front_frame.panels[0].display_payload.viewport == command.viewport
        assert board._selector_hold.display_payload.viewport == old_viewport
        assert board._require_selector_viewport(binding) == old_viewport

        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=finish)
        assert len(areas) == len(selections) == 1
        assert areas[0].viewport_revision == old_viewport.viewport_revision
        assert selections[0] is not None
        assert board._selector_hold is None
        assert board.visible_image_payload().viewport == command.viewport
    finally:
        harness.close()


def test_size_transition_waits_for_its_matching_plot_panel_front() -> None:
    """A size edit cannot stretch the accepted raster while Agg is in flight."""

    from gui_user_flow import close_task_console
    from test_task_console_figure_current import _image_value, _present_value, _wait
    from zlc_frontend.plot_layout import panel_surface_geometry
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(
                PanelConfig(
                    kind="2d",
                    title="Camera",
                    signal="image",
                    size="2x2",
                ),
            ),
        ),
        window_px=(1000, 800),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        _present_value(
            console,
            card,
            _image_value(revision=1, side=256, dtype=np.uint8),
            frame_key=("image", 1),
        )
        _wait(application, lambda: card.frozen_render_payload() is not None)

        old_front = card.board.front_frame
        old_size = (card.board.width(), card.board.height())
        card._on_size("4x4")

        # The request changed, but the old raster and its authored logical box
        # remain one accepted fact until the matching worker result exists.
        assert card.board.front_frame is old_front
        assert (card.board.width(), card.board.height()) == old_size

        _wait(application, lambda: card.board.front_frame is not old_front)
        geometry = panel_surface_geometry("4x4")
        front = card.board.front_frame
        assert (card.board.width(), card.board.height()) == geometry.logical_size
        assert (
            front.panels[0].raster.width,
            front.panels[0].raster.height,
        ) == geometry.raster_size
    finally:
        close_task_console(application, console)
