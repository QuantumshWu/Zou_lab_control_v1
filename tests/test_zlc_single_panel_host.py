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

import test_zlc_qt_image_interaction as image_fixtures


def _host_with_image_front():
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    host = SinglePanelHost("image", group="camera")
    host.resize(640, 420)
    host.show()
    host.present_frame(image_fixtures._frame(0))
    application.processEvents()
    return application, host


def test_present_frame_binds_the_image_family_and_forwards_typed_intents() -> None:
    """A composer-minted image frame lands on the host's board, creates the
    rectangle-selector binding gated by the host's Selectors switch, and the
    completed gestures come back as signals: the DISPLAY ONLY rectangle, the
    zoom commit's candidate viewport, and the clim commit's fixed limits."""

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import ImageColorLimitsCommit, RectangleGesture

    application, host = _host_with_image_front()
    try:
        board = host.board
        assert "image" in board._image_bindings
        # Selectors default OFF: the binding is READY (the just-presented
        # frame is the host's one source) but the board-wide switch is off.
        assert board._image_bindings["image"].interaction_ready is True
        assert board._selector_enabled is False

        rectangles: list[object] = []
        views: list[object] = []
        limits: list[object] = []
        host.rectangleSelected.connect(rectangles.append)
        host.viewCommitted.connect(views.append)
        host.colorLimitsCommitted.connect(limits.append)

        host.set_selectors_enabled(True)
        target = image_fixtures._target(board)
        before = board.visible_image_payload().viewport

        start = image_fixtures._point(target, 0.30, 0.30)
        end = image_fixtures._point(target, 0.70, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        image_fixtures._drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert rectangles and isinstance(rectangles[-1], RectangleGesture)

        image_fixtures._wheel(
            board, image_fixtures._point(target, 0.5, 0.5), -120)
        assert views, "wheel zoom did not forward a viewport commit"
        assert views[-1].viewport_revision > before.viewport_revision

        origin = board.visible_image_origin()
        host._on_intent(ImageColorLimitsCommit(origin, (1200.0, 2800.0)))
        assert limits and limits[-1] == (1200.0, 2800.0)
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
        with pytest.raises(ValueError):
            host.present_frame(image_fixtures._frame(0, panel_id="other"))
    finally:
        host.close()
        application.processEvents()
