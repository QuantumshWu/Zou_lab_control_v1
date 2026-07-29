"""SinglePanelHost owns the selector family ONCE for every one-panel window.

The host has two entrances -- ``present_panel`` for windows that render their
own picture and ``present_frame`` for frames a worker composer already minted
-- and both must land on the SAME QtRasterBoard with the SAME gesture binding.
The image family (a console 2d card, the capture viewer) binds through the
rectangle selector and forwards its typed intents as Qt signals, exactly like
the numeric three; a frame for a different panel is refused outright.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from figure_surface_fixtures import curve_panel, image_panel
from gui_user_flow import (
    drag_mouse_move,
    point_in_rect,
    raster_subrect,
    send_wheel,
)


def _image_frame(sequence: int):
    from zlc_frontend.render import BoardFrame

    return BoardFrame(
        "image-board",
        0,
        sequence,
        (image_panel(sequence),),
    )


def _host_with_image_front():
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    host = SinglePanelHost("image", group="camera")
    host.resize(640, 420)
    host.show()
    host.present_frame(_image_frame(0))
    application.processEvents()
    return application, host


def test_present_frame_binds_the_image_family_and_forwards_typed_intents() -> None:
    """A composer-minted image frame lands on the host's board, creates the
    rectangle-selector binding gated by the host's Selectors switch, and the
    completed gestures come back as signals: the DISPLAY ONLY rectangle, the
    zoom commit's candidate viewport, and the clim commit's fixed limits."""

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import (
        ImageColorLimitsCommit,
        ImageViewportCommit,
        RectangleGesture,
    )

    application, host = _host_with_image_front()
    try:
        board = host.board
        # Selectors default OFF: the binding is READY (the just-presented
        # frame is the host's one source) but the board-wide switch is off.
        assert not host.selectors_enabled

        rectangles: list[object] = []
        views: list[object] = []
        limits: list[object] = []
        host.rectangleSelected.connect(rectangles.append)
        host.viewCommitted.connect(views.append)
        host.colorLimitsCommitted.connect(limits.append)

        host.set_selectors_enabled(True)
        payload = host.front_frame.panels[0].display_payload
        target = raster_subrect(
            board.rect(), payload.raster_geometry.image_bounds
        )
        before = payload.viewport

        start = point_in_rect(target, 0.30, 0.30)
        end = point_in_rect(target, 0.70, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert rectangles and isinstance(rectangles[-1], RectangleGesture)

        rail = raster_subrect(
            board.rect(), payload.raster_geometry.distribution_bounds
        )
        low_handle = point_in_rect(rail, 0.5, 0.99)
        raised_low = point_in_rect(rail, 0.5, 0.75)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=low_handle,
        )
        drag_mouse_move(board, raised_low, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.LeftButton,
            pos=raised_low,
        )
        assert limits and isinstance(limits[-1], ImageColorLimitsCommit)
        assert limits[-1].origin == host.visible_interaction_origin()
        assert limits[-1].color_limits[0] > payload.color_limits[0]
        assert limits[-1].color_limits[1] == payload.color_limits[1]
        assert host.discard_pending_interaction(limits[-1].origin)

        send_wheel(board, point_in_rect(target, 0.5, 0.5), -120)
        assert views, "wheel zoom did not forward a viewport commit"
        assert isinstance(views[-1], ImageViewportCommit)
        assert views[-1].origin == host.visible_interaction_origin()
        assert views[-1].viewport.viewport_revision > before.viewport_revision
    finally:
        host.close()
        application.processEvents()


def test_present_frame_refuses_a_frame_for_another_panel() -> None:
    """The host is ONE panel: a coherent frame minted for a different panel id
    must be refused before it can desynchronise board layout and binding."""

    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    host = SinglePanelHost("image", group="camera")
    try:
        from zlc_frontend.render import BoardFrame

        with pytest.raises(ValueError):
            host.present_frame(
                BoardFrame(
                    "curve-board",
                    0,
                    0,
                    (curve_panel(0),),
                )
            )
    finally:
        host.close()
        application.processEvents()
