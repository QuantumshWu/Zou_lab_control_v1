"""Focused Qt contracts for one exact interactive HISTOGRAM front.

The board already has exact CURVE interaction coverage.  These oracles only
add the semantics earned by a second simultaneous numeric panel: frozen shared
bins on a log count axis, typed histogram intents, panel-local pending/holds,
and lifecycle isolation from the existing curve binding.
"""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _input(dataset_name: str, sequence: int):
    from zlc_data import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
    from zlc_frontend.figure import DatasetId, EvaluatedInput

    schema = {"curve": "c", "histogram": "d"}[dataset_name] * 64
    dataset_id = DatasetId(dataset_name)
    return EvaluatedInput(
        dataset_id,
        DatasetRevisionRef(
            BlockId(f"{dataset_name}-block"),
            StreamGenerationId(f"{dataset_name}-generation"),
            schema,
            DatasetRevision(sequence + 1),
        ),
    )


def _source_and_stamp(
    panel_id: str,
    evaluated_input,
    sequence: int,
    presentation,
):
    from zlc_frontend.render import CoherenceStamp, SourceIdentity

    source = SourceIdentity(
        evaluated_input.dataset_id,
        evaluated_input.ref.block_id,
        evaluated_input.ref.stream_generation,
        evaluated_input.ref.schema_fingerprint,
    )
    stamp = CoherenceStamp(
        "run",
        f"{panel_id}-epoch-{sequence}",
        f"{panel_id}-frame",
        evaluated_input.ref.schema_fingerprint,
        ({"curve": "a", "histogram": "b"}[panel_id]) * 64,
        (evaluated_input,),
        (presentation,),
    )
    return source, stamp


def _curve_panel(
    sequence: int,
    *,
    display_revision: int = 0,
    x_limits: tuple[float, float] = (-0.5, 3.5),
    offset: float = 0.0,
):
    from zlc_data import AxisId, MONITOR_HISTORY
    from zlc_frontend.curve_display import CurveViewportTransform
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedCurve, EvaluatedSeries
    from zlc_frontend.render import (
        CurvePanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        RasterBuffer,
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
    curve = EvaluatedCurve(
        axis,
        "count",
        np.asarray((0.0, 1.0, 2.0, 3.0)) + offset,
        np.asarray((True, True, True, True)),
    )
    viewport = CurveViewportTransform(
        axis,
        display_revision,
        (0.15, 0.10, 0.85, 0.90),
        x_limits,
        (-1.0 + offset, 4.0 + offset),
        (-0.15, 3.15),
    )
    payload = CurvePanelPayload(
        evaluated_input,
        viewport,
        (EvaluatedSeries((), curve),),
        ("rolling ROI",),
    )
    presentation = PanelPresentationIdentity(
        "curve",
        "curve-document",
        0,
        0,
        display_revision,
    )
    source, stamp = _source_and_stamp(
        "curve", evaluated_input, sequence, presentation
    )
    raster = RasterBuffer(
        240,
        120,
        bytes((20, 30, 40, 255)) * (240 * 120),
    )
    return PanelFrame("curve", "curve", source, stamp, raster, payload)


def _histogram_panel(
    sequence: int,
    *,
    display_revision: int = 0,
    x_limits: tuple[float, float] | None = None,
    count_limits: tuple[float, float] = (0.5, 20.0),
    relim_mode=None,
    bin_count: int = 5,
    labels: tuple[str, str] = ("ROI A", "ROI B"),
    variant: int = 0,
    thresholds: tuple[float, ...] = (),
):
    from zlc_frontend.figure import EvaluatedHistogram, EvaluatedSeries
    from zlc_frontend.histogram_display import (
        HistogramBinProjection,
        HistogramCountScale,
        HistogramViewportTransform,
    )
    from zlc_frontend.display_range import RelimMode
    from zlc_frontend.render import (
        HistogramPanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        RasterBuffer,
    )

    evaluated_input = _input("histogram", sequence)
    samples = (
        (
            np.asarray((-1.0, 0.5, 1.0, 1.5, 3.0))
            if variant == 0
            else np.asarray((-1.0, 0.5, 2.5, 3.0, 3.5))
        ),
        np.asarray((-1.5, 0.25, 1.25, 3.25)),
    )
    histograms = tuple(
        EvaluatedHistogram(values, (), 0, "photoelectron") for values in samples
    )
    projection = HistogramBinProjection(
        tuple(histogram.samples for histogram in histograms),
        bins=bin_count,
    )
    home_x_limits = (
        float(projection.bin_edges[0]),
        float(projection.bin_edges[-1]),
    )
    if relim_mode is None:
        relim_mode = RelimMode.TIGHT
    viewport = HistogramViewportTransform(
        display_revision,
        (0.15, 0.10, 0.85, 0.90),
        home_x_limits if x_limits is None else x_limits,
        count_limits,
        home_x_limits,
        HistogramCountScale.LOG,
        relim_mode,
        x_limits is None,
        projection.requested_bin_count,
    )
    payload = HistogramPanelPayload(
        evaluated_input,
        viewport,
        tuple(EvaluatedSeries((), value) for value in histograms),
        labels,
        projection,
        thresholds=thresholds,
    )
    presentation = PanelPresentationIdentity(
        "histogram",
        "histogram-document",
        0,
        0,
        display_revision,
    )
    source, stamp = _source_and_stamp(
        "histogram", evaluated_input, sequence, presentation
    )
    raster = RasterBuffer(
        240,
        120,
        bytes((45, 35, 25, 255)) * (240 * 120),
    )
    return PanelFrame(
        "histogram",
        "histogram",
        source,
        stamp,
        raster,
        payload,
    )


def _frame(
    sequence: int,
    *,
    curve_revision: int = 0,
    curve_x_limits: tuple[float, float] = (-0.5, 3.5),
    curve_offset: float = 0.0,
    histogram_revision: int = 0,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_count_limits: tuple[float, float] = (0.5, 20.0),
    histogram_relim_mode=None,
    histogram_bin_count: int = 5,
    histogram_labels: tuple[str, str] = ("ROI A", "ROI B"),
    histogram_variant: int = 0,
):
    from zlc_frontend.render import BoardFrame

    return BoardFrame(
        "numeric-board",
        0,
        sequence,
        (
            _curve_panel(
                sequence,
                display_revision=curve_revision,
                x_limits=curve_x_limits,
                offset=curve_offset,
            ),
            _histogram_panel(
                sequence,
                display_revision=histogram_revision,
                x_limits=histogram_x_limits,
                count_limits=histogram_count_limits,
                relim_mode=histogram_relim_mode,
                bin_count=histogram_bin_count,
                labels=histogram_labels,
                variant=histogram_variant,
            ),
        ),
    )


def _board(frame, curve_commands, histogram_commands):
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    board = QtRasterBoard(("curve", "histogram"), columns=2)
    board.resize(900, 320)
    board.show()
    board.present(frame)
    board.bind_curve_interaction("curve", curve_commands.append)
    board.bind_histogram_interaction("histogram", histogram_commands.append)
    application.processEvents()
    return application, board


def _numeric_target(board, kind: str, *, panel_id: str | None = None):
    binding = board._numeric_binding_for_kind(kind, panel_id=panel_id)
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


def _data_point(target, x: float, y: float):
    from PyQt5 import QtCore

    normalized = target.payload.viewport.data_to_widget_normalized(x, y)
    return QtCore.QPoint(
        int(round(target.bounds.x() + normalized[0] * target.bounds.width())),
        int(round(target.bounds.y() + normalized[1] * target.bounds.height())),
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


def _accepted_histogram_frame(sequence: int, command, **kwargs):
    accepted = _frame(
        sequence,
        histogram_revision=command.viewport.display_revision,
        histogram_x_limits=command.viewport.x_limits,
        **kwargs,
    )
    panel = accepted.panels[1]
    payload = replace(
        panel.display_payload,
        evaluated_input=command.origin.input_identity,
    )
    return replace(
        accepted,
        panels=(
            accepted.panels[0],
            replace(
                panel,
                source_identity=command.origin.source_identity,
                coherence_stamp=replace(
                    panel.coherence_stamp,
                    inputs=(command.origin.input_identity,),
                ),
                display_payload=payload,
            ),
        ),
    )


def test_curve_and_histogram_bind_simultaneously_with_locked_cross() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.histogram_display import HistogramCountScale
    from zlc_frontend.selector import CurveViewportCommit, HistogramViewportCommit

    curve_commands: list[object] = []
    histogram_commands: list[object] = []
    application, board = _board(_frame(1), curve_commands, histogram_commands)
    try:
        assert set(board._numeric_bindings) == {"curve", "histogram"}
        assert board._numeric_bindings["curve"].kind == "curve"
        histogram_binding = board._numeric_bindings["histogram"]
        assert histogram_binding.kind == "histogram"

        target = _numeric_target(board, "histogram")
        assert target is not None
        assert target.payload.viewport.count_scale is HistogramCountScale.LOG
        bin_index = int(np.argmax(target.payload.bin_counts[0]))
        bin_left = float(target.payload.bin_edges[bin_index])
        bin_right = float(target.payload.bin_edges[bin_index + 1])
        bin_count = int(target.payload.bin_counts[0][bin_index])
        cross_position = _data_point(
            target,
            0.5 * (bin_left + bin_right),
            float(bin_count),
        )

        exact_origin = board.visible_histogram_origin()
        assert exact_origin is not None
        assert _wheel(board, _point(target.plot, 0.5, 0.5), -120).isAccepted()
        histogram_zoom = histogram_commands[-1]
        assert isinstance(histogram_zoom, HistogramViewportCommit)
        assert histogram_zoom.origin == exact_origin
        assert histogram_zoom.origin.input_identity is target.payload.evaluated_input
        assert histogram_zoom.viewport.count_limits == target.payload.viewport.count_limits
        assert histogram_zoom.viewport.count_scale is HistogramCountScale.LOG
        assert curve_commands == []
        assert board.discard_pending_histogram_interaction(histogram_zoom.origin)

        curve_target = _numeric_target(board, "curve")
        assert curve_target is not None
        assert _wheel(board, _point(curve_target.plot, 0.5, 0.5), -120).isAccepted()
        assert isinstance(curve_commands[-1], CurveViewportCommit)
        assert len(histogram_commands) == 1
        assert board.discard_pending_curve_interaction(curve_commands[-1].origin)

        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=cross_position)
        assert histogram_binding.cross is not None
        _double_click(board, cross_position, QtCore.Qt.RightButton)
        assert histogram_binding.cross is None
    finally:
        board.close()
        application.processEvents()


def test_histogram_x_only_zoom_pan_area_home_and_exact_pending_discard() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import HistogramRangeGesture, HistogramViewportCommit

    commands: list[object] = []
    application, board = _board(_frame(1), [], commands)
    try:
        target = _numeric_target(board, "histogram")
        assert target is not None
        count_limits = target.payload.viewport.count_limits

        assert _wheel(board, _point(target.plot, 0.5, 0.5), -120).isAccepted()
        first_zoom = commands[-1]
        assert isinstance(first_zoom, HistogramViewportCommit)
        wrong_origin = replace(first_zoom.origin, sequence=first_zoom.origin.sequence + 1)
        assert not board.discard_pending_histogram_interaction(wrong_origin)
        assert board.discard_pending_histogram_interaction(first_zoom.origin)
        assert not board.discard_pending_histogram_interaction(first_zoom.origin)

        _wheel(board, _point(target.plot, 0.5, 0.5), -120)
        zoom = commands[-1]
        board.present(_accepted_histogram_frame(2, zoom))
        assert board._numeric_bindings["histogram"].pending_viewport is None
        assert zoom.viewport.count_limits == count_limits

        with pytest.raises(ValueError, match="stale histogram display revision"):
            board.present(_frame(3, histogram_revision=0))
        assert board.front_frame.sequence == 2
        assert board.histogram_selector_fault is None

        target = _numeric_target(board, "histogram")
        assert target is not None
        press = _point(target.plot, 0.5, 0.5)
        move = _point(target.plot, 0.6, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=press)
        _drag_move(board, move, QtCore.Qt.MiddleButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=move)
        pan = commands[-1]
        assert isinstance(pan, HistogramViewportCommit)
        assert pan.viewport.count_limits == count_limits
        assert pan.viewport.x_limits[0] < zoom.viewport.x_limits[0]
        board.present(_accepted_histogram_frame(4, pan))

        target = _numeric_target(board, "histogram")
        assert target is not None
        _double_click(board, _point(target.plot, 0.5, 0.5), QtCore.Qt.MiddleButton)
        home = commands[-1]
        assert isinstance(home, HistogramViewportCommit)
        assert home.viewport.x_limits == target.payload.viewport.home_x_limits
        assert home.viewport.count_limits == count_limits
        board.present(_accepted_histogram_frame(5, home))

        target = _numeric_target(board, "histogram")
        assert target is not None
        start = _point(target.plot, 0.25, 0.5)
        end = _point(target.plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        selected = commands[-1]
        assert isinstance(selected, HistogramRangeGesture)
        assert selected.origin == board.visible_histogram_origin()
        board.set_histogram_range_candidate(selected.x_span)
        # Click OUTSIDE the standing box: a press on its centre would MOVE it
        # (the reference's centre-handle grab), never clear it.
        outside = _point(target.plot, 0.05, 0.10)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=outside)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=outside)
        clear = commands[-1]
        assert isinstance(clear, HistogramRangeGesture) and clear.x_span is None
        board.set_histogram_range_candidate(clear.x_span)
        assert board._numeric_bindings["histogram"].applied_span is None
        board.set_histogram_range_candidate(selected.x_span)
        _double_click(board, _point(target.plot, 0.5, 0.5), QtCore.Qt.MiddleButton)
        area = commands[-1]
        assert isinstance(area, HistogramViewportCommit)
        assert area.viewport.x_limits == selected.x_span
        assert area.viewport.count_limits == count_limits
    finally:
        board.close()
        application.processEvents()


def test_histogram_hold_is_panel_local_while_curve_and_board_front_advance() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import HistogramRangeGesture

    commands: list[object] = []
    application, board = _board(_frame(1), [], commands)
    try:
        target = _numeric_target(board, "histogram")
        assert target is not None
        start = _point(target.plot, 0.25, 0.5)
        end = _point(target.plot, 0.75, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        held_payload = board.visible_histogram_payload()
        held_origin = board.visible_histogram_origin()
        held_pixels = board._selector_hold.prepared[0]
        assert held_payload is target.payload
        assert held_origin is not None and held_origin.sequence == 1

        board.present(
            _frame(2, curve_offset=10.0, histogram_variant=1)
        )
        assert board.front_frame.sequence == 2
        assert board.visible_histogram_payload() is held_payload
        assert board.visible_histogram_origin() == held_origin
        assert board._selector_hold.prepared[0] is held_pixels
        current_histogram = board.front_frame.panels[1].display_payload
        assert current_histogram is not held_payload
        curve = board.visible_curve_payload()
        assert curve is board.front_frame.panels[0].display_payload
        np.testing.assert_array_equal(
            curve.series[0].data.values,
            (10.0, 11.0, 12.0, 13.0),
        )

        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert isinstance(commands[-1], HistogramRangeGesture)
        assert commands[-1].origin == held_origin
        assert board._selector_hold is None
        assert board.visible_histogram_payload() is current_histogram
        assert board.visible_histogram_origin().sequence == 2
    finally:
        board.close()
        application.processEvents()


def test_same_revision_changes_only_data_derived_histogram_ranges() -> None:
    from zlc_frontend.display_range import RelimMode

    application, board = _board(_frame(1), [], [])
    try:
        board.present(
            _frame(
                2,
                histogram_variant=1,
                histogram_count_limits=(0.5, 40.0),
            )
        )
        assert board.front_frame.sequence == 2
    finally:
        board.close()
        application.processEvents()

    application, board = _board(
        _frame(
            3,
            histogram_x_limits=(-1.0, 3.0),
            histogram_count_limits=(0.5, 20.0),
            histogram_relim_mode=RelimMode.FIXED,
        ),
        [],
        [],
    )
    try:
        with pytest.raises(ValueError, match="conflicting authored state"):
            board.present(
                _frame(
                    4,
                    histogram_x_limits=(-0.5, 3.0),
                    histogram_count_limits=(0.5, 20.0),
                    histogram_relim_mode=RelimMode.FIXED,
                )
            )
        assert board.front_frame.sequence == 3
        with pytest.raises(ValueError, match="conflicting authored state"):
            board.present(
                _frame(
                    6,
                    histogram_x_limits=(-1.0, 3.0),
                    histogram_count_limits=(0.5, 20.0),
                    histogram_relim_mode=RelimMode.FIXED,
                    histogram_bin_count=6,
                )
            )
        assert board.front_frame.sequence == 3
        with pytest.raises(ValueError, match="conflicting authored state"):
            board.present(
                _frame(
                    5,
                    histogram_x_limits=(-1.0, 3.0),
                    histogram_count_limits=(0.5, 40.0),
                    histogram_relim_mode=RelimMode.FIXED,
                )
            )
        assert board.front_frame.sequence == 3
    finally:
        board.close()
        application.processEvents()


def test_histogram_readiness_and_callback_fault_do_not_disable_curve() -> None:
    from zlc_frontend.selector import CurveViewportCommit, HistogramViewportCommit

    curve_commands: list[object] = []
    histogram_commands: list[object] = []
    application, board = _board(_frame(1), curve_commands, histogram_commands)
    try:
        board.set_interaction_readiness(
            image=False,
            curve=True,
            histogram=False,
        )
        histogram_target = _numeric_target(board, "histogram")
        curve_target = _numeric_target(board, "curve")
        assert histogram_target is not None and curve_target is not None
        stale = _wheel(board, _point(histogram_target.plot, 0.5, 0.5), -120)
        assert not stale.isAccepted()
        assert histogram_commands == []
        assert board.histogram_selector_fault is None
        assert board._numeric_bindings["histogram"].binding_enabled

        assert _wheel(board, _point(curve_target.plot, 0.5, 0.5), -120).isAccepted()
        assert isinstance(curve_commands[-1], CurveViewportCommit)
        assert board.discard_pending_curve_interaction(curve_commands[-1].origin)

        board.set_interaction_readiness(
            image=False,
            curve=True,
            histogram=True,
        )
        board.bind_histogram_interaction(
            "histogram",
            lambda _command: (_ for _ in ()).throw(RuntimeError("histogram boom")),
        )
        histogram_target = _numeric_target(board, "histogram")
        assert histogram_target is not None
        assert _wheel(
            board, _point(histogram_target.plot, 0.5, 0.5), -120
        ).isAccepted()
        assert board.histogram_selector_fault is not None
        assert not board._numeric_bindings["histogram"].binding_enabled
        assert board.curve_selector_fault is None
        assert board._numeric_bindings["curve"].binding_enabled

        assert _wheel(board, _point(curve_target.plot, 0.5, 0.5), -120).isAccepted()
        assert isinstance(curve_commands[-1], CurveViewportCommit)
        assert not isinstance(curve_commands[-1], HistogramViewportCommit)
    finally:
        board.close()
        application.processEvents()


def test_histogram_hold_releases_on_escape_hide_semantics_and_unbind() -> None:
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

    application, board = _board(_frame(1), [], [])
    try:
        target = _numeric_target(board, "histogram")
        assert target is not None
        start = _point(target.plot, 0.2, 0.5)

        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.keyClick(board, QtCore.Qt.Key_Escape)
        assert board._selector_hold is None
        assert board._numeric_bindings["histogram"].span_candidate is None

        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        assert board._selector_hold is not None
        QtWidgets.QApplication.sendEvent(board, QtGui.QHideEvent())
        assert board._selector_hold is None

        board.show()
        application.processEvents()
        target = _numeric_target(board, "histogram")
        assert target is not None
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target.plot, 0.2, 0.5),
        )
        board.present(
            _frame(2, histogram_labels=("Renamed ROI", "ROI B"))
        )
        assert board._selector_hold is None
        assert board._numeric_bindings["curve"].binding_enabled
        assert board.curve_selector_fault is None

        target = _numeric_target(board, "histogram")
        assert target is not None
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target.plot, 0.2, 0.5),
        )
        assert board._selector_hold is not None
        board.unbind_histogram_interaction()
        assert board._selector_hold is None
        assert "histogram" not in board._numeric_bindings
        assert "curve" in board._numeric_bindings
        assert board.visible_curve_payload() is not None
    finally:
        board.close()
        application.processEvents()


def test_histogram_threshold_line_drag_is_live_exclusive_and_near_line_only() -> None:
    """The design's frozen histogram selector row: a left press within 2% of
    the x span of an authored threshold line grabs THAT line (the area
    machinery never starts), every motion commits the authored set live,
    release only ends the drag, and a press away from any line falls through
    to the ordinary area pull."""

    from PyQt5 import QtCore, QtGui, QtTest
    from zlc_frontend.render import BoardFrame
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app
    from zlc_frontend.selector import HistogramThresholdCommit

    application = ensure_qt_app()
    commands: list[object] = []
    board = QtRasterBoard(("histogram",), columns=1)
    board.resize(640, 420)
    board.show()
    board.present(BoardFrame(
        "numeric-board", 0, 0,
        (_histogram_panel(1, thresholds=(1.0,)),),
    ))
    board.bind_histogram_interaction("histogram", commands.append)
    application.processEvents()
    try:
        binding = board._numeric_bindings["histogram"]
        target = _numeric_target(board, "histogram")
        assert target is not None
        x_low, x_high = target.payload.viewport.x_limits
        line_fraction = (1.0 - x_low) / (x_high - x_low)
        on_line = _point(target.plot, line_fraction, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=on_line)
        assert binding.threshold_drag == 0
        # EXCLUSIVE with the area selector: no span machinery started.
        assert binding.rectangle_drag is None and binding.span_rect is None

        drag_fraction = line_fraction + 0.2
        expected = x_low + drag_fraction * (x_high - x_low)
        board.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(_point(target.plot, drag_fraction, 0.5)),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        application.processEvents()
        command = commands[-1]
        assert isinstance(command, HistogramThresholdCommit)
        assert command.thresholds == pytest.approx((expected,), abs=0.05)
        first_answer = binding.threshold_pending_answer
        assert first_answer is not None and first_answer.origin == command.origin

        # The still-painted raster retains the press-time threshold.  Returning
        # to it is a newer desired state, but the exact first answer remains the
        # sole in-flight command; pointer input coalesces into one render-paced
        # mailbox rather than overwriting the answer that can admit a frame.
        board.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(on_line),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        application.processEvents()
        assert len(commands) == 1
        assert binding.threshold_pending_answer is first_answer
        assert binding.queued_thresholds == pytest.approx((1.0,), abs=0.05)

        answered_panel = _histogram_panel(
            2,
            display_revision=first_answer.display_revision,
            thresholds=first_answer.thresholds,
        )
        answered_panel = replace(
            answered_panel,
            source_identity=first_answer.origin.source_identity,
            coherence_stamp=replace(
                answered_panel.coherence_stamp,
                inputs=(first_answer.origin.input_identity,),
            ),
            display_payload=replace(
                answered_panel.display_payload,
                evaluated_input=first_answer.origin.input_identity,
            ),
        )
        board.present(BoardFrame(
            "numeric-board",
            0,
            1,
            (answered_panel,),
        ))
        returned = commands[-1]
        assert isinstance(returned, HistogramThresholdCommit)
        assert returned.thresholds == pytest.approx((1.0,), abs=0.05)
        assert len(commands) == 2
        pending_answer = binding.threshold_pending_answer
        assert pending_answer is not None
        assert pending_answer.thresholds == returned.thresholds
        assert pending_answer.display_revision > first_answer.display_revision
        assert binding.queued_thresholds is None
        issued = len(commands)

        QtTest.QTest.mouseRelease(
            board, QtCore.Qt.LeftButton,
            pos=on_line)
        # Release only ends the drag; the last step is not re-issued.
        assert len(commands) == issued
        assert binding.threshold_drag is None
        assert board._selector_hold is None
        assert binding.threshold_pending_answer is pending_answer
        assert board.discard_pending_histogram_interaction(returned.origin)
        assert binding.threshold_pending_answer is None

        # A press AWAY from any line (far past the 2% tolerance) starts the
        # ordinary area pull instead.
        away = _point(target.plot, line_fraction - 0.4, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=away)
        assert binding.threshold_drag is None
        assert binding.rectangle_drag is not None
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=away)
    finally:
        board.close()
        application.processEvents()
