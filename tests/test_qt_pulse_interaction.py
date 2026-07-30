"""PULSE is a first-class member of the unified board interaction family.

The pulse preview presents on the SAME QtRasterBoard as every other panel kind
and its gestures run through the SAME numeric interaction owner: area select,
wheel zoom and pan speak the CURVE intent vocabulary over a PulsePanelPayload
whose viewport is the shared NumericViewportTransform.  This is the design's
"one selector owner, no second family" rule made mechanical: the payload rides
the real ``render_pulse_timeline_panel`` output, the gestures are driven the
way a person drives them, and every intent must come back typed and x-only.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui_user_flow import (
    drag_mouse_move,
    normalized_subrect,
    point_in_rect,
    send_mouse_double_click,
    send_wheel,
)


def _pulse_panel(*, digest: str = "e" * 64):
    from zlc_frontend.matplotlib_render import render_pulse_timeline_panel
    from zlc_frontend.render import (
        DocumentInputIdentity,
        PanelFrame,
    )

    document_input = DocumentInputIdentity("pulse-document", 1, digest)
    raster, payload = render_pulse_timeline_panel(
        pulses=[dict(channel="ch00", start=0.0, stop=1e-3, name="cool"),
                dict(channel="ch01", start=2e-4, stop=8e-4, name="probe")],
        channels=["ch00", "ch01"],
        channel_labels={"ch00": "cooling", "ch01": "probe"},
        total_duration=2e-3,
        title="preview",
        size="2x2",
        document_input=document_input,
    )
    return PanelFrame(
        "pulse", "pulse", document_input, None, raster, payload
    )


def _pulse_frame(*, panel=None, sequence: int = 0):
    from zlc_frontend.render import BoardFrame

    return BoardFrame(
        "pulse-board",
        0,
        sequence,
        (_pulse_panel() if panel is None else panel,),
    )


def _pulse_host(events, *, panel=None):
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app

    application = ensure_qt_app()
    host = SinglePanelHost("pulse")
    host.resize(600, 450)
    host.show()
    host.set_selectors_enabled(True)
    host.rangeSelected.connect(events.append)
    host.viewCommitted.connect(events.append)
    host.crossSelected.connect(events.append)
    host.present_frame(_pulse_frame(panel=panel))
    application.processEvents()
    return application, host


def _pulse_plot_rect(host):
    frame = host.front_frame
    assert frame is not None
    payload = frame.panels[0].display_payload
    return normalized_subrect(host.board.rect(), payload.viewport.plot_bounds)


def test_pulse_area_select_is_display_only_and_cannot_resolve_a_selection() -> None:
    """Left drag on the pulse preview emits a CurveRangeGesture whose x_span is
    a real TIME span inside the drawn frame.  It remains display-only: pulse
    documents have no dataset axis on which an authority Selection can exist.
    A degenerate click clears the span.
    """

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture

    events: list[object] = []
    application, host = _pulse_host(events)
    board = host.board
    try:
        plot = _pulse_plot_rect(host)
        payload = host.front_frame.panels[0].display_payload
        x_low, x_high = payload.viewport.x_limits
        origin = host.visible_interaction_origin()
        assert origin is not None

        start = point_in_rect(plot, 0.25, 0.50)
        end = point_in_rect(plot, 0.75, 0.50)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)

        gesture = events[-1]
        assert isinstance(gesture, CurveRangeGesture)
        assert gesture.origin == origin
        assert gesture.x_span is not None
        span_low, span_high = gesture.x_span
        assert x_low < span_low < span_high < x_high, (
            f"the selected span {gesture.x_span} must be a time range inside "
            f"the drawn frame {payload.viewport.x_limits}")
        assert not hasattr(payload.viewport.x_axis, "axis_id")
        with pytest.raises(TypeError, match="requires a dataset source"):
            host.area_commit_for_range_gesture(gesture, figure=object())

        # Click OUTSIDE the standing box (a press on its edge/centre is the
        # reference's resize/move grab, not a clearing click).
        outside = point_in_rect(plot, 0.05, 0.10)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=outside)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=outside)
        clear = events[-1]
        assert isinstance(clear, CurveRangeGesture) and clear.x_span is None
        assert host.area_commit_for_range_gesture(clear, figure=object()) is None
    finally:
        host.close()
        application.processEvents()


def test_pulse_wheel_zoom_is_x_only_and_typed() -> None:
    """Wheel zoom over the pulse preview emits a CurveViewportCommit whose
    candidate changes ONLY the x limits (time); the row axis never moves.
    """

    from zlc_frontend.selector import CurveViewportCommit

    events: list[object] = []
    application, host = _pulse_host(events)
    board = host.board
    try:
        plot = _pulse_plot_rect(host)
        payload = host.front_frame.panels[0].display_payload
        before = payload.viewport

        assert send_wheel(
            board,
            point_in_rect(plot, 0.5, 0.5),
            -120,
        ).isAccepted()
        command = events[-1]
        assert isinstance(command, CurveViewportCommit)
        assert command.origin == host.visible_interaction_origin()
        candidate = command.viewport
        assert candidate.x_limits != before.x_limits, "wheel did not zoom time"
        assert candidate.y_limits == before.y_limits, (
            "pulse zoom must be x-only -- the row axis moved")
        assert candidate.display_revision == before.display_revision + 1
        assert candidate.home_x_limits == before.home_x_limits, (
            "home must stay pinned to the drawn frame")
    finally:
        host.close()
        application.processEvents()


def test_pulse_cross_pins_and_clears() -> None:
    """Right click pins a continuous cross; right double-click clears it."""

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CrossGesture

    events: list[object] = []
    application, host = _pulse_host(events)
    board = host.board
    try:
        plot = _pulse_plot_rect(host)

        position = point_in_rect(plot, 0.33, 0.61)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=position)
        pinned = events[-1]
        assert isinstance(pinned, CrossGesture)
        assert pinned.origin == host.visible_interaction_origin()
        assert pinned.point is not None
        payload = host.front_frame.panels[0].display_payload
        x_low, x_high = payload.viewport.x_limits
        assert x_low <= pinned.point[0] <= x_high

        send_mouse_double_click(board, position, QtCore.Qt.RightButton)
        cleared = events[-1]
        assert isinstance(cleared, CrossGesture)
        assert cleared.point is None
    finally:
        host.close()
        application.processEvents()


def test_pulse_area_box_is_data_anchored_and_double_middle_zooms_to_it() -> None:
    """The reference keeps its selector artists in DATA coordinates: after an
    area select, a double-middle re-xlims the view to EXACTLY the selection,
    and the standing box (still the same data span) now covers the whole
    zoomed view instead of clinging to stale screen fractions."""

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import CurveRangeGesture, CurveViewportCommit

    events: list[object] = []
    application, host = _pulse_host(events)
    board = host.board
    try:
        plot = _pulse_plot_rect(host)
        before_y_limits = (
            host.front_frame.panels[0].display_payload.viewport.y_limits
        )
        start = point_in_rect(plot, 0.30, 0.30)
        end = point_in_rect(plot, 0.60, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        gesture = events[-1]
        assert isinstance(gesture, CurveRangeGesture)
        span = gesture.x_span
        assert span is not None

        send_mouse_double_click(
            board,
            point_in_rect(plot, 0.5, 0.5),
            QtCore.Qt.MiddleButton,
        )
        command = events[-1]
        assert isinstance(command, CurveViewportCommit)
        assert command.viewport.x_limits == pytest.approx(span)
        assert command.viewport.y_limits == before_y_limits
    finally:
        host.close()
        application.processEvents()


def test_pulse_panel_is_strictly_document_backed_and_rejects_mixed_identity() -> None:
    """No dataset/run/join identity may leak into a pulse document front."""

    from dataclasses import replace

    from zlc_data import (
        BlockId,
        DatasetRevision,
        DatasetRevisionRef,
        StreamGenerationId,
    )
    from zlc_frontend.figure import DatasetId, EvaluatedInput
    from zlc_frontend.render import (
        BoardFrame,
        CoherenceStamp,
        DocumentInputIdentity,
        PanelFrame,
        PulsePanelPayload,
        SourceIdentity,
    )
    from zlc_frontend.selector import PanelInteractionOrigin

    panel = _pulse_panel()
    document = panel.source_identity
    assert isinstance(document, DocumentInputIdentity)
    assert panel.coherence_stamp is None
    assert isinstance(panel.display_payload, PulsePanelPayload)
    for forbidden in (
        "dataset_id", "block_id", "stream_generation", "run_id",
        "join_key_digest", "inputs",
    ):
        assert not hasattr(document, forbidden)
    assert not hasattr(panel.display_payload.viewport.x_axis, "axis_id")

    changed = DocumentInputIdentity(
        document.document_id, document.document_revision, "f" * 64
    )
    with pytest.raises(ValueError, match="payload differs"):
        PanelFrame(
            panel.panel_id,
            panel.coherence_group,
            document,
            None,
            panel.raster,
            replace(panel.display_payload, document_input=changed),
        )
    dataset_source = SourceIdentity(
        DatasetId("not-a-pulse-document"),
        BlockId("block"),
        StreamGenerationId("generation"),
        "b" * 64,
    )
    dataset_ref = DatasetRevisionRef(
        dataset_source.block_id,
        dataset_source.stream_generation,
        dataset_source.schema_fingerprint,
        DatasetRevision(1),
    )
    dataset_input = EvaluatedInput(dataset_source.dataset_id, dataset_ref)
    dataset_stamp = CoherenceStamp((dataset_input,))
    with pytest.raises(TypeError, match="PulsePanelPayload requires"):
        PanelFrame(
            panel.panel_id,
            panel.coherence_group,
            dataset_source,
            dataset_stamp,
            panel.raster,
            panel.display_payload,
        )
    document_board = BoardFrame("document-board", 0, 0, (panel,))
    with pytest.raises(TypeError, match="document interaction"):
        PanelInteractionOrigin(
            document_board,
            panel,
            dataset_input,
            panel.display_payload.viewport.display_revision,
        )
    dataset_panel = PanelFrame(
        "dataset", "mixed", dataset_source, dataset_stamp, panel.raster
    )
    with pytest.raises(ValueError, match="one exact CoherenceStamp"):
        BoardFrame(
            "board",
            0,
            0,
            (dataset_panel, replace(panel, coherence_group="mixed")),
        )


def test_same_document_revision_with_changed_digest_makes_old_gesture_stale() -> None:
    """The CAS includes content digest, not only a reusable revision label."""

    from dataclasses import replace

    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import DocumentInputIdentity
    from zlc_frontend.selector import CurveViewportCommit

    application = ensure_qt_app()
    host = SinglePanelHost("pulse")
    first = _pulse_panel()
    host.present_panel(first.raster, first.display_payload)
    stale = host.visible_interaction_origin()
    assert stale is not None
    assert host.front_frame.panels[0].coherence_stamp is None

    current_document = DocumentInputIdentity(
        first.source_identity.document_id,
        first.source_identity.document_revision,
        "f" * 64,
    )
    current_payload = replace(
        first.display_payload,
        document_input=current_document,
    )
    host.present_panel(first.raster, current_payload)
    current = host.visible_interaction_origin()
    assert current is not None and current != stale
    commit = CurveViewportCommit(
        stale,
        replace(
            first.display_payload.viewport,
            display_revision=first.display_payload.viewport.display_revision + 1,
            x_limits=(2e-4, 8e-4),
        ),
    )
    assert commit.origin != current
    assert commit.origin.input_identity.content_digest == "e" * 64
    assert current.input_identity.content_digest == "f" * 64
    host.close()
    application.processEvents()


def test_document_digest_change_cancels_a_real_inflight_pulse_gesture() -> None:
    """A drag cannot commit against a newer document with a reused revision."""

    from PyQt5 import QtCore, QtTest

    events: list[object] = []
    application, host = _pulse_host(events)
    board = host.board
    try:
        plot = _pulse_plot_rect(host)
        start = point_in_rect(plot, 0.25, 0.5)
        end = point_in_rect(plot, 0.75, 0.5)
        old_origin = host.visible_interaction_origin()
        assert old_origin is not None
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        drag_mouse_move(board, end, QtCore.Qt.LeftButton)

        changed = _pulse_panel(digest="f" * 64)
        host.present_frame(_pulse_frame(panel=changed, sequence=1))
        assert host.visible_interaction_origin() != old_origin
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert events == []
    finally:
        host.close()
        application.processEvents()
