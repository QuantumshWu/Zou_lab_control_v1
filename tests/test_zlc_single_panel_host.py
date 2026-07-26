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

import test_u02c_qt_curve_interaction as image_fixtures


def _image_frame(sequence: int):
    from zlc_frontend.render import BoardFrame

    return BoardFrame(
        "image-board",
        0,
        sequence,
        (image_fixtures._image_panel(sequence),),
    )


def _image_target(board):
    return board._selector_target(board._image_bindings["image"])[0]


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
        target = _image_target(board)
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
        assert isinstance(views[-1], ImageViewportCommit)
        assert views[-1].origin == board.visible_image_origin()
        assert views[-1].viewport.viewport_revision > before.viewport_revision

        origin = board.visible_image_origin()
        color_commit = ImageColorLimitsCommit(origin, (1200.0, 2800.0))
        host._on_intent(color_commit)
        assert limits and limits[-1] is color_commit
        assert limits[-1].color_limits == (1200.0, 2800.0)
    finally:
        host.close()
        application.processEvents()


def test_live_color_limits_track_the_latest_authored_value_until_answered() -> None:
    """A live H drag may return to the still-painted limits before Agg answers.

    The return is a newer desired state, but only one exact answer may be in
    flight; the latest value is issued from the admitted frame immediately
    after that answer arrives.
    """

    from zlc_frontend.qt_widgets._raster_image_interaction import (
        _held_panel_from_target,
    )
    from dataclasses import replace

    from zlc_frontend.render import BoardFrame
    from zlc_frontend.selector import ImageColorLimitsCommit

    application, host = _host_with_image_front()
    try:
        host.set_selectors_enabled(True)
        board = host.board
        binding = board._image_bindings["image"]
        target = board._selector_target(binding)
        assert target is not None
        hold = _held_panel_from_target(target)
        board._selector_hold = hold

        commits: list[ImageColorLimitsCommit] = []
        host.colorLimitsCommitted.connect(commits.append)
        painted = board.visible_image_payload().color_limits
        span = painted[1] - painted[0]
        changed = (painted[0] + 0.1 * span, painted[1] - 0.1 * span)

        assert board._commit_color_limits(binding, changed, hold=hold)
        first_answer = binding.pending_color_answer
        assert first_answer is not None
        assert first_answer.color_limits == changed

        assert board._commit_color_limits(binding, painted, hold=hold)
        assert binding.pending_color_answer is first_answer
        assert binding.queued_color_limits == painted
        assert [commit.color_limits for commit in commits] == [changed]

        answered = image_fixtures._image_panel(
            1,
            viewport_revision=first_answer.display_revision,
        )
        answered = replace(
            answered,
            source_identity=first_answer.origin.source_identity,
            coherence_stamp=replace(
                answered.coherence_stamp,
                inputs=(first_answer.origin.input_identity,),
            ),
            display_payload=replace(
                answered.display_payload,
                evaluated_input=first_answer.origin.input_identity,
                color_limits=changed,
            ),
        )
        host.present_frame(BoardFrame("image-board", 0, 1, (answered,)))
        latest_answer = binding.pending_color_answer
        assert latest_answer is not None
        assert latest_answer.color_limits == painted
        assert latest_answer.display_revision > first_answer.display_revision
        assert binding.queued_color_limits is None
        assert [commit.color_limits for commit in commits] == [changed, painted]

        board._cancel_image_gesture(binding, clear_draft=False)
        assert board._selector_hold is None
        assert binding.pending_color_answer is latest_answer
        assert board.discard_pending_image_interaction(commits[-1].origin)
        assert binding.pending_color_answer is None
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
                    (image_fixtures._curve_panel(0),),
                )
            )
    finally:
        host.close()
        application.processEvents()
