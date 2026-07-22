"""PULSE is a first-class member of the unified board interaction family.

The pulse preview presents on the SAME QtRasterBoard as every other panel kind
and its gestures run through the SAME numeric interaction owner: area select,
wheel zoom and pan speak the CURVE intent vocabulary over a PulsePanelPayload
whose viewport is the shared CurveViewportTransform.  This is the design's
"one selector owner, no second family" rule made mechanical: the payload rides
the real ``render_pulse_timeline_panel`` output, the gestures are driven the
way a person drives them, and every intent must come back typed and x-only.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pulse_panel():
    from zlc_data import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
    from zlc_frontend.figure import DatasetId, EvaluatedInput
    from zlc_frontend.matplotlib_render import render_pulse_timeline_panel
    from zlc_frontend.render import (
        CoherenceStamp,
        PanelFrame,
        PanelPresentationIdentity,
        SourceIdentity,
    )

    ref = DatasetRevisionRef(
        BlockId("pulse-block"),
        StreamGenerationId("pulse-generation"),
        "e" * 64,
        DatasetRevision(1),
    )
    evaluated_input = EvaluatedInput(DatasetId("pulse"), ref)
    raster, payload = render_pulse_timeline_panel(
        pulses=[dict(channel="ch00", start=0.0, stop=1e-3, name="cool"),
                dict(channel="ch01", start=2e-4, stop=8e-4, name="probe")],
        channels=["ch00", "ch01"],
        channel_labels={"ch00": "cooling", "ch01": "probe"},
        total_duration=2e-3,
        title="preview",
        size="2x2",
        evaluated_input=evaluated_input,
    )
    presentation = PanelPresentationIdentity(
        "pulse", "pulse-document", 0, 0, payload.viewport.display_revision
    )
    stamp = CoherenceStamp(
        "run",
        "pulse-epoch-0",
        "pulse-frame",
        ref.schema_fingerprint,
        "f" * 64,
        (evaluated_input,),
        (presentation,),
    )
    source = SourceIdentity(
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )
    return PanelFrame("pulse", "pulse", source, stamp, raster, payload)


def _board(commands):
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    application = ensure_qt_app()
    board = QtRasterBoard(("pulse",), columns=1)
    board.resize(600, 450)
    board.show()
    board.present(BoardFrame("pulse-board", 0, 0, (_pulse_panel(),)))
    board.bind_pulse_interaction("pulse", commands.append)
    application.processEvents()
    return application, board


def _target(board):
    binding = board._numeric_binding_for_kind("pulse")
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


def test_pulse_area_select_is_a_time_span_and_resolves_a_selection() -> None:
    """Left drag on the pulse preview emits a CurveRangeGesture whose x_span is
    a real TIME span inside the drawn frame, resolvable to a Selection on the
    time axis while the hold is alive; a degenerate click clears the span.
    """

    from PyQt5 import QtCore, QtTest
    from zlc_data import Selection
    from zlc_frontend.selector import CurveRangeGesture

    commands: list[object] = []
    selections: list[Selection] = []
    application, board = _board(commands)

    def resolve(gesture):
        commands.append(gesture)
        if isinstance(gesture, CurveRangeGesture) and gesture.x_span is not None:
            selections.append(board.selection_for_curve_range_gesture(gesture))

    board.unbind_pulse_interaction()
    board.bind_pulse_interaction("pulse", resolve)
    try:
        plot = _target(board).plot
        payload = board.visible_pulse_payload()
        assert payload is not None
        x_low, x_high = payload.viewport.x_limits

        start, end = _point(plot, 0.25, 0.50), _point(plot, 0.75, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, end, QtCore.Qt.LeftButton)
        assert board._selector_hold is not None, "area drag must hold the panel front"
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)

        gesture = commands[-1]
        assert isinstance(gesture, CurveRangeGesture)
        assert gesture.x_span is not None
        span_low, span_high = gesture.x_span
        assert x_low < span_low < span_high < x_high, (
            f"the selected span {gesture.x_span} must be a time range inside "
            f"the drawn frame {payload.viewport.x_limits}")
        assert selections, "the completed span did not resolve to a Selection"
        board.set_pulse_range_candidate(gesture.x_span)
        assert board._numeric_bindings["pulse"].applied_span == pytest.approx(
            gesture.x_span)

        # Click OUTSIDE the standing box (a press on its edge/centre is the
        # reference's resize/move grab, not a clearing click).
        outside = _point(plot, 0.05, 0.10)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=outside)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=outside)
        clear = commands[-1]
        assert isinstance(clear, CurveRangeGesture) and clear.x_span is None
        board.set_pulse_range_candidate(clear.x_span)
        assert board._numeric_bindings["pulse"].applied_span is None
    finally:
        board.close()
        application.processEvents()


def test_pulse_wheel_zoom_is_x_only_and_typed() -> None:
    """Wheel zoom over the pulse preview emits a CurveViewportCommit whose
    candidate changes ONLY the x limits (time); the row axis never moves.
    """

    from zlc_frontend.selector import CurveViewportCommit

    commands: list[object] = []
    application, board = _board(commands)
    try:
        plot = _target(board).plot
        payload = board.visible_pulse_payload()
        assert payload is not None
        before = payload.viewport

        _wheel(board, _point(plot, 0.5, 0.5), -120)
        command = commands[-1]
        assert isinstance(command, CurveViewportCommit)
        candidate = command.viewport
        assert candidate.x_limits != before.x_limits, "wheel did not zoom time"
        assert candidate.y_limits == before.y_limits, (
            "pulse zoom must be x-only -- the row axis moved")
        assert candidate.display_revision == before.display_revision + 1
        assert candidate.home_x_limits == before.home_x_limits, (
            "home must stay pinned to the drawn frame")
    finally:
        board.close()
        application.processEvents()


def test_pulse_cross_pins_and_clears_and_hover_never_snaps() -> None:
    """Right click pins a continuous cross readout; right double-click clears
    it.  Pointer motion must never raise and never fabricates a hover sample --
    a pulse timeline has rows, not sampled series.
    """

    from PyQt5 import QtCore, QtGui, QtTest

    commands: list[object] = []
    application, board = _board(commands)
    try:
        plot = _target(board).plot
        binding = board._numeric_bindings["pulse"]

        QtTest.QTest.mouseMove(board, _point(plot, 0.40, 0.30))
        application.processEvents()
        assert binding.hover is None, "a pulse row is not a sample to snap to"

        position = _point(plot, 0.33, 0.61)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=position)
        cross = binding.cross
        assert cross is not None, "right click must pin the continuous cross"
        payload = board.visible_pulse_payload()
        x_low, x_high = payload.viewport.x_limits
        assert x_low <= cross.x <= x_high

        board.mouseDoubleClickEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QPointF(position),
                QtCore.Qt.RightButton,
                QtCore.Qt.RightButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert binding.cross is None, "right double-click must clear the cross"
    finally:
        board.close()
        application.processEvents()
