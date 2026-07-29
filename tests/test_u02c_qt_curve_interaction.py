"""Real Qt curve interaction through the current single-panel owner."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from figure_surface_fixtures import curve_panel
from gui_user_flow import (
    drag_mouse_move,
    normalized_subrect,
    point_in_rect,
    send_mouse_double_click,
    send_wheel,
)


def _curve_frame(
    sequence: int,
    *,
    display_revision: int = 0,
    offset: float = 0.0,
    panel=None,
):
    from zlc_frontend.render import BoardFrame

    if panel is None:
        panel = curve_panel(
            sequence,
            display_revision=display_revision,
            offset=offset,
        )
    return BoardFrame("curve-board", 0, sequence, (panel,))


def _curve_host(frame):
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    host = SinglePanelHost("curve")
    host.resize(400, 300)
    host.show()
    host.set_selectors_enabled(True)
    host.present_frame(frame)
    application.processEvents()
    return application, host


def _curve_plot_rect(host):
    frame = host.front_frame
    assert frame is not None
    payload = frame.panels[0].display_payload
    return normalized_subrect(host.board.rect(), payload.viewport.plot_bounds)


def _accepted_curve_frame(sequence: int, command):
    panel = curve_panel(
        sequence,
        display_revision=command.viewport.display_revision,
    )
    payload = replace(
        panel.display_payload,
        evaluated_input=command.origin.input_identity,
        viewport=command.viewport,
    )
    return _curve_frame(
        sequence,
        panel=replace(
            panel,
            source_identity=command.origin.source_identity,
            coherence_stamp=replace(
                panel.coherence_stamp,
                inputs=(command.origin.input_identity,),
            ),
            display_payload=payload,
        ),
    )


def test_curve_uses_draw_frozen_bbox_and_horizontal_span() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture

    application, host = _curve_host(_curve_frame(1))
    ranges: list[object] = []
    host.rangeSelected.connect(ranges.append)
    board = host.board
    try:
        plot = _curve_plot_rect(host)
        assert plot.left() == 80.0
        assert plot.top() == 30.0
        assert plot.width() == pytest.approx(240.0)
        assert plot.height() == pytest.approx(240.0)

        start = point_in_rect(plot, 0.25, 0.50)
        end = point_in_rect(plot, 0.75, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)

        selected = ranges[-1]
        assert isinstance(selected, CurveRangeGesture)
        assert selected.x_span == pytest.approx((0.5, 2.5))
        assert selected.origin == host.visible_interaction_origin()

        # A degenerate click outside the standing box clears that display-only
        # candidate.  SinglePanelHost echoes both selection and clear through
        # the same current range owner before emitting the typed gesture.
        outside = point_in_rect(plot, 0.05, 0.10)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=outside)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=outside)
        cleared = ranges[-1]
        assert isinstance(cleared, CurveRangeGesture)
        assert cleared.x_span is None
    finally:
        host.close()
        application.processEvents()


def test_curve_continuous_cross_pins_and_clears() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CrossGesture

    application, host = _curve_host(_curve_frame(1))
    crosses: list[object] = []
    host.crossSelected.connect(crosses.append)
    board = host.board
    try:
        arbitrary = point_in_rect(_curve_plot_rect(host), 0.33, 0.61)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=arbitrary)
        pinned = crosses[-1]
        assert isinstance(pinned, CrossGesture)
        assert pinned.origin == host.visible_interaction_origin()
        assert pinned.point is not None
        assert pinned.point[0] != round(pinned.point[0])

        send_mouse_double_click(board, arbitrary, QtCore.Qt.RightButton)
        cleared = crosses[-1]
        assert isinstance(cleared, CrossGesture)
        assert cleared.point is None
    finally:
        host.close()
        application.processEvents()


def test_curve_x_only_wheel_pan_and_area_home() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture, CurveViewportCommit

    application, host = _curve_host(_curve_frame(1))
    views: list[object] = []
    ranges: list[object] = []
    host.viewCommitted.connect(views.append)
    host.rangeSelected.connect(ranges.append)
    board = host.board
    try:
        plot = _curve_plot_rect(host)
        viewport = host.front_frame.panels[0].display_payload.viewport
        center = point_in_rect(plot, 0.5, 0.5)
        initial_y = viewport.y_limits
        assert send_wheel(board, center, -120).isAccepted()
        zoom = views[-1]
        assert isinstance(zoom, CurveViewportCommit)
        assert zoom.viewport.y_limits == initial_y
        assert zoom.viewport.x_limits == pytest.approx(
            (-0.3181818182, 3.3181818182)
        )

        # Each next intent is issued only after the exact accepted front is
        # presented again through the public host owner.
        host.present_frame(_accepted_curve_frame(2, zoom))
        plot = _curve_plot_rect(host)
        viewport = host.front_frame.panels[0].display_payload.viewport
        press = point_in_rect(plot, 0.50, 0.50)
        move = point_in_rect(plot, 0.60, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=press)
        drag_mouse_move(board, move, QtCore.Qt.MiddleButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=move)
        pan = views[-1]
        assert isinstance(pan, CurveViewportCommit)
        assert pan.viewport.y_limits == viewport.y_limits
        assert pan.viewport.x_limits[0] < zoom.viewport.x_limits[0]

        host.present_frame(_accepted_curve_frame(3, pan))
        plot = _curve_plot_rect(host)
        viewport = host.front_frame.panels[0].display_payload.viewport
        send_mouse_double_click(
            board,
            point_in_rect(plot, 0.5, 0.5),
            QtCore.Qt.MiddleButton,
        )
        home = views[-1]
        assert isinstance(home, CurveViewportCommit)
        assert home.viewport.x_limits == viewport.home_x_limits
        assert home.viewport.y_limits == viewport.y_limits

        host.present_frame(_accepted_curve_frame(4, home))
        plot = _curve_plot_rect(host)
        viewport = host.front_frame.panels[0].display_payload.viewport
        start = point_in_rect(plot, 0.25, 0.5)
        end = point_in_rect(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        selected = ranges[-1]
        assert isinstance(selected, CurveRangeGesture)
        assert selected.x_span is not None

        send_mouse_double_click(
            board,
            point_in_rect(plot, 0.5, 0.5),
            QtCore.Qt.MiddleButton,
        )
        area = views[-1]
        assert isinstance(area, CurveViewportCommit)
        assert area.viewport.x_limits == selected.x_span
        assert area.viewport.y_limits == viewport.y_limits
    finally:
        host.close()
        application.processEvents()


def test_curve_hold_keeps_the_press_front_while_current_front_advances() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture

    application, host = _curve_host(_curve_frame(1))
    ranges: list[object] = []
    host.rangeSelected.connect(ranges.append)
    board = host.board
    try:
        old_origin = host.visible_interaction_origin()
        assert old_origin is not None
        plot = _curve_plot_rect(host)
        start = point_in_rect(plot, 0.25, 0.5)
        end = point_in_rect(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)

        host.present_frame(_curve_frame(2, offset=10.0))
        assert host.front_frame.sequence == 2
        assert host.visible_interaction_origin() == old_origin

        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        selected = ranges[-1]
        assert isinstance(selected, CurveRangeGesture)
        assert selected.origin == old_origin
        assert selected.x_span == pytest.approx((0.5, 2.5))
        assert host.visible_interaction_origin() != old_origin
    finally:
        host.close()
        application.processEvents()


def test_curve_host_discards_only_the_exact_pending_view() -> None:
    from zlc_frontend.selector import CurveViewportCommit

    application, host = _curve_host(_curve_frame(1))
    views: list[object] = []
    host.viewCommitted.connect(views.append)
    board = host.board
    try:
        center = point_in_rect(_curve_plot_rect(host), 0.5, 0.5)
        send_wheel(board, center, -120)
        first = views[-1]
        assert isinstance(first, CurveViewportCommit)
        assert host.discard_pending_interaction(first.origin)
        assert not host.discard_pending_interaction(first.origin)

        send_wheel(board, center, -120)
        assert len(views) == 2
        assert isinstance(views[-1], CurveViewportCommit)
    finally:
        host.close()
        application.processEvents()


def test_curve_host_readiness_parks_input_without_fault() -> None:
    from zlc_frontend.selector import CurveViewportCommit

    application, host = _curve_host(_curve_frame(1))
    views: list[object] = []
    host.viewCommitted.connect(views.append)
    board = host.board
    try:
        center = point_in_rect(_curve_plot_rect(host), 0.5, 0.5)
        host.set_interaction_ready(False)
        parked = send_wheel(board, center, -120)
        assert not parked.isAccepted()
        assert views == []
        assert host.selector_fault is None

        host.set_interaction_ready(True)
        assert send_wheel(board, center, -120).isAccepted()
        assert isinstance(views[-1], CurveViewportCommit)
        assert host.selector_fault is None
    finally:
        host.close()
        application.processEvents()


def test_extreme_curve_transform_is_rejected_inside_qt_event_boundary() -> None:
    from PyQt5 import QtCore, QtTest

    panel = curve_panel(1)
    panel = replace(
        panel,
        display_payload=replace(
            panel.display_payload,
            viewport=replace(
                panel.display_payload.viewport,
                x_limits=(-8.0e307, 9.0e307),
            ),
        ),
    )
    application, host = _curve_host(_curve_frame(1, panel=panel))
    views: list[object] = []
    host.viewCommitted.connect(views.append)
    board = host.board
    try:
        plot = _curve_plot_rect(host)
        assert send_wheel(
            board,
            point_in_rect(plot, 0.5, 0.5),
            120,
        ).isAccepted()
        assert views == []
        assert host.selector_fault is None

        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.MiddleButton,
            pos=point_in_rect(plot, 0.25, 0.5),
        )
        overflow_point = point_in_rect(plot, 2.0, 0.5)
        drag_mouse_move(board, overflow_point, QtCore.Qt.MiddleButton)
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=overflow_point,
        )
        assert views == []
        assert host.selector_fault is None
    finally:
        host.close()
        application.processEvents()


def test_curve_host_unbind_cancels_active_input_and_rebinds_on_present() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture

    application, host = _curve_host(_curve_frame(1))
    ranges: list[object] = []
    host.rangeSelected.connect(ranges.append)
    board = host.board
    try:
        plot = _curve_plot_rect(host)
        start = point_in_rect(plot, 0.25, 0.5)
        end = point_in_rect(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        host.unbind_interaction()
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert ranges == []
        assert host.visible_interaction_origin() is None

        host.present_frame(_curve_frame(2, display_revision=1))
        plot = _curve_plot_rect(host)
        start = point_in_rect(plot, 0.25, 0.5)
        end = point_in_rect(plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert isinstance(ranges[-1], CurveRangeGesture)
        assert host.visible_interaction_origin() is not None
    finally:
        host.close()
        application.processEvents()


def test_qt_selector_art_tokens_delegate_to_the_frontend_owner() -> None:
    from zlc_frontend.qt_widgets.style import (
        SELECTOR_ALPHA,
        SELECTOR_COLOR,
        SELECTOR_DOT_PX,
        SELECTOR_FONT_PX,
        SELECTOR_HANDLE_PX,
        SELECTOR_LINE_PX,
    )
    from zlc_frontend.selector_visual import (
        SELECTOR_ALPHA as OWNER_ALPHA,
        SELECTOR_COLOR as OWNER_COLOR,
        SELECTOR_DOT_PX as OWNER_DOT_PX,
        SELECTOR_FONT_PX as OWNER_FONT_PX,
        SELECTOR_HANDLE_PX as OWNER_HANDLE_PX,
        SELECTOR_LINE_PX as OWNER_LINE_PX,
    )

    assert (
        SELECTOR_ALPHA,
        SELECTOR_COLOR,
        SELECTOR_DOT_PX,
        SELECTOR_FONT_PX,
        SELECTOR_HANDLE_PX,
        SELECTOR_LINE_PX,
    ) == (
        OWNER_ALPHA,
        OWNER_COLOR,
        OWNER_DOT_PX,
        OWNER_FONT_PX,
        OWNER_HANDLE_PX,
        OWNER_LINE_PX,
    )
