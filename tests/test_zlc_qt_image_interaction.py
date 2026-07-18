from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _viewport(
    *,
    revision: int = 0,
    bounds=(0.0, 0.0, 1.0, 1.0),
    width: int = 2,
    height: int = 2,
):
    from zlc_data import (
        AxisId,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.image_view import ImageViewportTransform

    coordinate_frame = CoordinateFrameId("qt-image-interaction")
    return ImageViewportTransform(
        (
            AxisSpec(
                AxisId("camera.y"),
                "y",
                SPATIAL_Y,
                height,
                tuple(20.0 + 10.0 * index for index in range(height)),
                unit="pixel",
                coordinate_frame=coordinate_frame,
            ),
            AxisSpec(
                AxisId("camera.x"),
                "x",
                SPATIAL_X,
                width,
                tuple(10.0 + index for index in range(width)),
                unit="pixel",
                coordinate_frame=coordinate_frame,
            ),
        ),
        viewport_revision=revision,
        visible_bounds=bounds,
    )


def _frame(
    sequence: int,
    *,
    viewport=None,
    values=None,
    document_revision: int = 0,
    source_suffix: str = "",
    color_limits: tuple[float, float] = (1500.0, 3500.0),
):
    from zlc_data import (
        BlockId,
        DatasetRevision,
        DatasetRevisionRef,
        StreamGenerationId,
    )
    from zlc_frontend.image_display import ImageDisplayState, ImageRelimMode
    from zlc_frontend.figure import DatasetId, EvaluatedAxis, EvaluatedImage, EvaluatedInput
    from zlc_frontend.image_raster import rasterize_image_indexed8
    from zlc_frontend.render import (
        BoardFrame,
        CoherenceStamp,
        ImagePanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        SourceIdentity,
    )

    viewport = _viewport() if viewport is None else viewport
    values = (
        np.asarray([[1000.0, 2000.0], [np.nan, 4000.0]], dtype=np.float64)
        if values is None
        else np.asarray(values, dtype=np.float64)
    )
    image = EvaluatedImage(
        EvaluatedAxis(
            viewport.x_axis.axis_id,
            viewport.x_axis.name,
            viewport.x_axis.unit,
            tuple(range(viewport.x_axis.size)),
            tuple(viewport.x_axis.coordinates),
        ),
        EvaluatedAxis(
            viewport.y_axis.axis_id,
            viewport.y_axis.name,
            viewport.y_axis.unit,
            tuple(range(viewport.y_axis.size)),
            tuple(viewport.y_axis.coordinates),
        ),
        values,
        np.ones(values.shape, dtype=bool),
    )
    display = ImageDisplayState(
        revision=viewport.viewport_revision,
        relim_mode=ImageRelimMode.FIXED,
        fixed_color_limits=color_limits,
    )
    raster, data_range, histogram, accepted_limits = rasterize_image_indexed8(
        image,
        display,
        current_color_limits=None,
        previous_relim_mode=None,
    )
    assert accepted_limits == color_limits
    schema = "a" * 64
    dataset_id = DatasetId("camera")
    ref = DatasetRevisionRef(
        BlockId(f"camera-block{source_suffix}"),
        StreamGenerationId(f"camera-generation{source_suffix}"),
        schema,
        DatasetRevision(sequence + 1),
    )
    evaluated_input = EvaluatedInput(dataset_id, ref)
    # A deliberately non-grey codebook makes accidental fallback LUTs obvious.
    palette = tuple(
        0xFF000000 | (index << 16) | ((255 - index) << 8) | (index // 2)
        for index in range(256)
    )
    payload = ImagePanelPayload(
        image=image,
        evaluated_input=evaluated_input,
        viewport=viewport,
        data_range=data_range,
        histogram_counts=histogram,
        base_palette=palette,
        color_limits=color_limits,
    )
    source = SourceIdentity(
        dataset_id,
        ref.block_id,
        ref.stream_generation,
        schema,
    )
    presentation = PanelPresentationIdentity(
        "image",
        "camera-document",
        document_revision,
        0,
        viewport.viewport_revision,
    )
    stamp = CoherenceStamp(
        "camera-run",
        f"epoch-{sequence}",
        "camera-frame",
        schema,
        "b" * 64,
        (evaluated_input,),
        (presentation,),
    )
    return BoardFrame(
        "camera-board",
        0,
        sequence,
        (PanelFrame("image", "camera", source, stamp, raster, payload),),
    )


def _target(board):
    return board._selector_target()[0]


def _point(target, x_fraction: float, y_fraction: float):
    from PyQt5 import QtCore

    return QtCore.QPoint(
        target.left() + int(x_fraction * max(1, target.width() - 1)),
        target.top() + int(y_fraction * max(1, target.height() - 1)),
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


def _mouse_press(board, position, button) -> None:
    from PyQt5 import QtCore, QtGui

    board.mousePressEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            QtCore.QPointF(position),
            button,
            button,
            QtCore.Qt.NoModifier,
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("panel_id", "   ", ValueError),
        ("board_id", "", ValueError),
        ("layout_generation", -1, ValueError),
        ("sequence", -1, ValueError),
        ("viewport_revision", -1, ValueError),
        ("source_identity", object(), TypeError),
        ("normalized_bounds", (0.8, 0.0, 0.2, 1.0), ValueError),
    ),
)
def test_rectangle_gesture_rejects_invalid_exact_origin(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    from zlc_frontend.selector import RectangleGesture

    source = _frame(0).panels[0].source_identity
    values = {
        "panel_id": "image",
        "board_id": "camera-board",
        "layout_generation": 0,
        "sequence": 1,
        "source_identity": source,
        "normalized_bounds": (0.0, 0.0, 1.0, 1.0),
        "viewport_revision": 0,
    }
    values[field] = value
    with pytest.raises(error):
        RectangleGesture(**values)


def _bound_board(frame, *, interaction_callback=None, selection_callback=None):
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    board = QtRasterBoard(("image",), columns=1)
    board.resize(240, 180)
    board.show()
    board.present(frame)
    board.bind_rectangle_selector(
        "image",
        frame.panels[0].image_payload.viewport,
        (lambda _gesture: None) if selection_callback is None else selection_callback,
        enabled=True,
        interaction_callback=interaction_callback,
    )
    application.processEvents()
    return application, board


def test_indexed_lut_and_exact_cross_never_infer_z_from_display_code() -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    frame = _frame(1)
    application, board = _bound_board(frame)
    try:
        payload = frame.panels[0].image_payload
        assert payload is not None
        qimage = board._front[1][0][1]
        assert tuple(qimage.colorTable()) == payload.base_palette

        position = _point(_target(board), 0.75, 0.25)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=position)
        sample = board._cross_sample
        assert sample is not None
        assert (sample.x_index, sample.y_index) == (1, 0)
        assert (sample.x_coordinate, sample.y_coordinate) == (11.0, 20.0)
        assert sample.value == 2000.0 and sample.valid
        display_code = frame.panels[0].raster.pixels[1]
        assert display_code != sample.value

        invalid_position = _point(_target(board), 0.25, 0.75)
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=invalid_position,
        )
        invalid = board._cross_sample
        assert invalid is not None and np.isnan(invalid.value) and not invalid.valid
    finally:
        board.close()
        application.processEvents()


def test_indexed_front_without_codebook_payload_fails_closed() -> None:
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    frame = _frame(1)
    bare = replace(
        frame,
        panels=(replace(frame.panels[0], image_payload=None),),
    )
    board = QtRasterBoard(("image",), columns=1)
    try:
        with pytest.raises(ValueError, match="requires ImagePanelPayload"):
            board.present(bare)
        assert not board.has_front
    finally:
        board.close()
        application.processEvents()


def test_exact_interaction_callback_rejects_current_or_future_payloadless_panel() -> None:
    from zlc_frontend.render import PixelFormat, RasterBuffer
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    exact = _frame(1)
    gray = replace(
        exact,
        panels=(
            replace(
                exact.panels[0],
                raster=RasterBuffer(2, 2, 2, PixelFormat.GRAY8, bytes((1, 2, 3, 4))),
                image_payload=None,
            ),
        ),
    )
    viewport = exact.panels[0].image_payload.viewport
    current = QtRasterBoard(("image",), columns=1)
    future = QtRasterBoard(("image",), columns=1)
    try:
        current.present(gray)
        with pytest.raises(ValueError, match="requires exact ImagePanelPayload"):
            current.bind_rectangle_selector(
                "image",
                viewport,
                lambda _gesture: None,
                interaction_callback=lambda _command: None,
            )
        # The legacy A-only seam remains available for a payloadless panel.
        current.bind_rectangle_selector(
            "image",
            viewport,
            lambda _gesture: None,
        )

        future.bind_rectangle_selector(
            "image",
            viewport,
            lambda _gesture: None,
            interaction_callback=lambda _command: None,
        )
        with pytest.raises(ValueError, match="requires exact ImagePanelPayload"):
            future.present(gray)
        assert not future.has_front
    finally:
        current.close()
        future.close()
        application.processEvents()


def test_hover_is_exact_and_ephemeral_on_leave_disable_and_hide() -> None:
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

    application, board = _bound_board(_frame(1))
    try:
        position = _point(_target(board), 0.75, 0.75)
        QtTest.QTest.mouseMove(board, position)
        sample = board._hover_sample
        assert sample is not None and sample.value == 4000.0 and sample.valid

        # A stationary pointer remains useful on a live image: every promoted
        # front re-evaluates it against the new exact payload/provenance.
        board.present(_frame(2, values=[[1.0, 2.0], [3.0, 8.0]]))
        application.processEvents()
        sample = board._hover_sample
        assert sample is not None and sample.value == 8.0 and sample.valid

        QtWidgets.QApplication.sendEvent(board, QtCore.QEvent(QtCore.QEvent.Leave))
        assert board._hover_sample is None
        QtTest.QTest.mouseMove(board, QtCore.QPoint(position.x() - 1, position.y()))
        QtTest.QTest.mouseMove(board, position)
        assert board._hover_sample is not None
        board.set_rectangle_selector_enabled(False)
        assert board._hover_sample is None
        assert not _wheel(board, position, -120).isAccepted()

        board.set_rectangle_selector_enabled(True)
        QtTest.QTest.mouseMove(board, QtCore.QPoint(position.x() - 1, position.y()))
        QtTest.QTest.mouseMove(board, position)
        assert board._hover_sample is not None
        QtWidgets.QApplication.sendEvent(board, QtGui.QHideEvent())
        assert board._hover_sample is None
    finally:
        board.close()
        application.processEvents()


def test_area_drag_exposes_and_persists_snapped_endpoint_label_from_visible_axis_precision(
    monkeypatch,
) -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    application, board = _bound_board(_frame(1))
    try:
        target = _target(board)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.05, 0.05),
        )
        _drag_move(board, _point(target, 0.95, 0.95), QtCore.Qt.LeftButton)
        bounds = board._selector_draft_bounds
        assert bounds is not None
        label = board._selection_endpoint_label(bounds)
        assert label.count("\n") == 1
        assert label.startswith("(") and ")\n(" in label and label.endswith(")")
        # Visible spans are x=1 and y=10, matching main's span/1000 rule.
        first, second = label.splitlines()
        assert "." in first and "." in second
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.95, 0.95),
        )
        selection = board._require_selector_viewport().selection_for_normalized_bounds(
            bounds
        )
        board.set_selector_applied_selection(selection)
        board.set_selector_draft_selection(None)

        painted_endpoint_bounds = []
        original = board._paint_selector_rectangle

        def record_endpoint(*args, **kwargs):
            painted_endpoint_bounds.append(kwargs["endpoint_bounds"])
            return original(*args, **kwargs)

        monkeypatch.setattr(board, "_paint_selector_rectangle", record_endpoint)
        canvas = QtGui.QImage(board.size(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(canvas)
        try:
            board._paint_selector_overlays(painter)
        finally:
            painter.end()
        assert bounds in painted_endpoint_bounds
    finally:
        board.close()
        application.processEvents()


@pytest.mark.parametrize(("width", "height"), ((1, 2), (2, 1)))
def test_selection_endpoint_label_supports_singleton_spatial_axes(
    width: int,
    height: int,
) -> None:
    viewport = _viewport(width=width, height=height)
    values = np.arange(width * height, dtype=np.float64).reshape(height, width)
    application, board = _bound_board(_frame(1, viewport=viewport, values=values))
    try:
        label = board._selection_endpoint_label((0.0, 0.0, 1.0, 1.0))
        first, second = label.splitlines()
        assert first.startswith("(") and second.startswith("(")
        if width == 1:
            assert first.split(",")[0] == second.split(",")[0]
        if height == 1:
            assert first.split(",")[1] == second.split(",")[1]
    finally:
        board.close()
        application.processEvents()


def test_selection_endpoint_label_survives_a_non_cell_aligned_zoom() -> None:
    from unittest.mock import Mock

    viewport = _viewport().centered_zoom(
        (0.33, 0.61),
        1.0 / 1.1,
        viewport_revision=1,
    )
    application, board = _bound_board(_frame(1, viewport=viewport))
    try:
        bounds = viewport.snapped_bounds_for_drag((0.1, 0.1), (0.9, 0.9))
        label = board._selection_endpoint_label(bounds)
        assert label.count("\n") == 1
        board.set_selector_draft_selection(
            viewport.selection_for_normalized_bounds(bounds)
        )
        painter = Mock()
        painter.fontMetrics.return_value = board.fontMetrics()
        board._paint_selector_overlays(painter)
        assert painter.drawText.called
    finally:
        board.close()
        application.processEvents()


def test_color_rail_maps_physical_values_through_committed_limits() -> None:
    from zlc_frontend.qt_widgets import QtRasterBoard

    frame = _frame(
        1,
        values=[[1000.0, 2250.0], [3000.0, 4000.0]],
        color_limits=(2000.0, 3000.0),
    )
    payload = frame.panels[0].image_payload
    assert payload is not None
    domain = QtRasterBoard._color_rail_domain(payload)
    assert domain == pytest.approx((1920.0, 3080.0))
    assert payload.data_range == (1000.0, 4000.0)
    assert QtRasterBoard._color_rail_argb(payload, 1000.0) == payload.base_palette[1]
    assert QtRasterBoard._color_rail_argb(payload, 2000.0) == payload.base_palette[1]
    assert QtRasterBoard._color_rail_argb(payload, 2500.0) == payload.base_palette[128]
    assert QtRasterBoard._color_rail_argb(payload, 3000.0) == payload.base_palette[255]
    assert QtRasterBoard._color_rail_argb(payload, 4000.0) == payload.base_palette[255]
    quarter_code = frame.panels[0].raster.pixels[1]
    assert quarter_code == 64
    assert QtRasterBoard._color_rail_argb(payload, 2250.0) == payload.base_palette[
        quarter_code
    ]


def test_wheel_down_zooms_in_once_and_up_zooms_out() -> None:
    committed = []
    first = _frame(1)
    application, board = _bound_board(first, interaction_callback=committed.append)
    try:
        position = _point(_target(board), 0.5, 0.5)
        _wheel(board, position, -120)
        assert len(committed) == 1
        zoomed = committed[-1].viewport
        assert zoomed.visible_bounds[2] - zoomed.visible_bounds[0] < 1.0
        # A second event cannot author a conflicting transform at the same
        # pending revision before the owner returns the committed front.
        _wheel(board, position, -120)
        assert len(committed) == 1

        board.present(_frame(2, viewport=zoomed))
        _wheel(board, position, 120)
        assert len(committed) == 2
        zoomed_out = committed[-1].viewport
        assert (
            zoomed_out.visible_bounds[2] - zoomed_out.visible_bounds[0]
            > zoomed.visible_bounds[2] - zoomed.visible_bounds[0]
        )
    finally:
        board.close()
        application.processEvents()


def test_one_active_gesture_blocks_chords_and_pending_only_gates_image_hit() -> None:
    from PyQt5 import QtCore, QtTest

    commands = []
    gestures = []
    application, board = _bound_board(
        _frame(1),
        interaction_callback=commands.append,
        selection_callback=gestures.append,
    )
    try:
        target = _target(board)
        start = _point(target, 0.1, 0.1)
        end = _point(target, 0.8, 0.8)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        hold = board._selector_hold
        anchor = board._selector_drag_anchor
        assert hold is not None and anchor is not None

        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        _mouse_press(board, _point(target, 0.5, 0.5), QtCore.Qt.MiddleButton)
        _mouse_press(board, _point(target, 0.5, 0.5), QtCore.Qt.RightButton)
        assert commands == [] and board._cross_sample is None
        assert board._selector_hold is hold
        assert board._selector_drag_anchor == anchor
        assert board._pan_anchor is None and board._clim_drag is None

        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert len(gestures) == 1 and commands == []

        target = _target(board)
        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        assert len(commands) == 1 and board._interaction_is_pending()
        # The pending IMAGE CAS is local to the image/rail. Black margin (or a
        # sibling cell in a larger board) must still bubble to its parent.
        assert not _wheel(board, QtCore.QPoint(1, 1), -120).isAccepted()
        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        assert len(commands) == 1
    finally:
        board.close()
        application.processEvents()


def test_middle_pan_uses_press_pixels_holds_only_exact_payload_and_commits_on_release() -> None:
    from PyQt5 import QtCore, QtTest

    initial_viewport = _viewport(revision=3, bounds=(0.2, 0.2, 0.8, 0.8))
    first = _frame(1, viewport=initial_viewport)
    committed = []
    application, board = _bound_board(first, interaction_callback=committed.append)
    try:
        target = _target(board)
        start = _point(target, 0.5, 0.5)
        end = QtCore.QPoint(start.x() + 24, start.y() + 9)
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=start)
        hold = board._selector_hold
        assert hold is not None
        assert hold.image_payload is first.panels[0].image_payload
        assert not isinstance(hold.image_payload, type(first))
        held_origin = board.visible_image_origin()
        assert held_origin is not None and held_origin.sequence == 1
        assert held_origin.evaluated_input is first.panels[0].image_payload.evaluated_input

        # The complete board advances, while the interacting panel retains the
        # exact press front and transform only.
        board.present(
            _frame(
                2,
                viewport=initial_viewport,
                values=[[9.0, 8.0], [7.0, 6.0]],
            )
        )
        assert board.front_frame.sequence == 2
        assert board._selector_hold is hold
        assert board.visible_image_origin() == held_origin
        _drag_move(board, end, QtCore.Qt.MiddleButton)
        assert committed == []
        QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=end)

        assert len(committed) == 1
        expected = initial_viewport.panned_by_pixels(
            (24.0, 9.0),
            (target.width(), target.height()),
        )
        assert committed[0].viewport == expected
        assert board._selector_hold is None
        assert board.visible_image_origin().sequence == 2
    finally:
        board.close()
        application.processEvents()


def test_middle_double_click_prefers_area_then_home_and_right_double_clears_cross() -> None:
    from PyQt5 import QtCore, QtTest

    committed = []
    first = _frame(1)
    application, board = _bound_board(first, interaction_callback=committed.append)
    try:
        viewport = first.panels[0].image_payload.viewport
        area = viewport.selection_for_normalized_bounds((0.0, 0.0, 0.5, 0.5))
        board.set_selector_applied_selection(area)
        target = _target(board)
        center = _point(target, 0.5, 0.5)
        QtTest.QTest.mouseDClick(board, QtCore.Qt.MiddleButton, pos=center)
        assert committed[-1].viewport.visible_bounds == (0.0, 0.0, 0.5, 0.5)

        board.present(_frame(2, viewport=committed[-1].viewport))
        board.set_selector_applied_selection(None)
        QtTest.QTest.mouseDClick(board, QtCore.Qt.MiddleButton, pos=center)
        assert committed[-1].viewport.visible_bounds == (0.0, 0.0, 1.0, 1.0)

        board.present(_frame(3, viewport=committed[-1].viewport))
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
        assert board._cross_sample is not None
        QtTest.QTest.mouseDClick(board, QtCore.Qt.RightButton, pos=center)
        assert board._cross_sample is None
    finally:
        board.close()
        application.processEvents()


def test_clim_handle_holds_exact_front_and_commits_without_a_temporary_lut() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_frontend.selector import ImageColorLimitsCommit

    committed = []
    first = _frame(1)
    application, board = _bound_board(first, interaction_callback=committed.append)
    try:
        rail, *_rest, payload = board._clim_rail_target()
        original_lut = tuple(board._front[1][0][1].colorTable())
        domain = board._color_rail_domain(payload)
        start = QtCore.QPoint(
            rail.center().x(),
            int(round(board._rail_y(payload.color_limits[0], domain, rail))),
        )
        end = QtCore.QPoint(
            start.x(),
            int(round(board._rail_y(2000.0, domain, rail))),
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        assert board.visible_image_payload() is payload
        assert board._selector_hold.image_payload is payload
        _drag_move(board, end, QtCore.Qt.LeftButton)
        assert committed == []
        assert tuple(board._front[1][0][1].colorTable()) == original_lut
        candidate_label = board._clim_candidate_label(payload)
        assert candidate_label.startswith("H low=")
        assert "high=3500" in candidate_label
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)

        assert len(committed) == 1
        command = committed[0]
        assert isinstance(command, ImageColorLimitsCommit)
        assert command.origin.presentation.panel_revision == 0
        assert command.color_limits[0] == pytest.approx(2000.0, abs=30.0)
        assert command.color_limits[1] == 3500.0
        assert board._selector_hold is None

        # Only the owner's new panel revision changes worker-quantized pixels
        # and releases the pending gate.  The one sampled cmap palette is
        # independent of clim and therefore stays byte-for-byte identical.
        original_pixels = first.panels[0].raster.pixels
        next_viewport = _viewport(revision=1)
        board.present(
            _frame(
                2,
                viewport=next_viewport,
                color_limits=command.color_limits,
            )
        )
        assert not board._interaction_is_pending()
        assert tuple(board._front[1][0][1].colorTable()) == original_lut
        assert board.front_frame.panels[0].raster.pixels != original_pixels

        # Every non-release lifecycle exit uses the same hold cleanup.
        rail, *_rest, payload = board._clim_rail_target()
        domain = board._color_rail_domain(payload)
        start = QtCore.QPoint(
            rail.center().x(),
            int(round(board._rail_y(payload.color_limits[0], domain, rail))),
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        assert board._selector_hold is not None
        board.set_rectangle_selector_enabled(False)
        assert board._selector_hold is None and board._clim_drag is None
    finally:
        board.close()
        application.processEvents()


def test_hold_badge_does_not_cover_a_locked_cross_value() -> None:
    from PyQt5 import QtCore, QtTest

    class RecordingPainter:
        def __init__(self, metrics):
            self._metrics = metrics
            self.filled = []

        def save(self):
            pass

        def restore(self):
            pass

        def setClipRect(self, _rect):
            pass

        def setPen(self, _pen):
            pass

        def setBrush(self, _brush):
            pass

        def drawLine(self, *_args):
            pass

        def drawEllipse(self, *_args):
            pass

        def drawText(self, *_args):
            pass

        def fontMetrics(self):
            return self._metrics

        def fillRect(self, rect, _color):
            self.filled.append(QtCore.QRect(rect))

    application, board = _bound_board(_frame(1))
    try:
        target = _target(board)
        center = _point(target, 0.5, 0.5)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
        assert board._cross_sample is not None
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.1, 0.1),
        )
        assert board._selector_hold is not None

        hold_painter = RecordingPainter(board.fontMetrics())
        board._paint_hold_badge(
            hold_painter,
            board._selector_hold,
            target,
            live_sequence=board.front_frame.sequence + 1,
        )
        cross_painter = RecordingPainter(board.fontMetrics())
        board._paint_cross_sample(
            cross_painter,
            board._cross_sample,
            target,
        )
        assert len(hold_painter.filled) == len(cross_painter.filled) == 1
        assert not hold_painter.filled[0].intersects(cross_painter.filled[0])
    finally:
        board.close()
        application.processEvents()


def test_locked_cross_value_remains_visible_when_zoom_moves_point_off_view() -> None:
    from unittest.mock import Mock

    from PyQt5 import QtCore, QtTest
    from zlc_frontend.image_view import ImageViewportTransform

    first = _frame(1)
    application, board = _bound_board(first)
    try:
        target = _target(board)
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=_point(target, 0.75, 0.75),
        )
        sample = board._cross_sample
        assert sample is not None
        viewport = ImageViewportTransform(
            first.panels[0].image_payload.viewport.axes,
            1,
            (0.0, 0.0, 0.5, 0.5),
        )
        board.present(_frame(2, viewport=viewport))
        assert board._cross_sample is sample

        painter = Mock()
        painter.fontMetrics.return_value = board.fontMetrics()
        board._paint_cross_sample(painter, sample, _target(board))
        painter.drawLine.assert_not_called()
        painter.drawEllipse.assert_not_called()
        painter.fillRect.assert_called_once()
        assert "off-view" in painter.drawText.call_args.args[-1]

        QtTest.QTest.mouseDClick(
            board,
            QtCore.Qt.RightButton,
            pos=_point(_target(board), 0.5, 0.5),
        )
        assert board._cross_sample is None
    finally:
        board.close()
        application.processEvents()


def test_terminal_fault_discards_only_the_exact_pending_origin() -> None:
    committed = []
    application, board = _bound_board(
        _frame(1),
        interaction_callback=committed.append,
    )
    try:
        _wheel(board, _point(_target(board), 0.5, 0.5), -120)
        origin = committed[0].origin
        assert origin.evaluated_input is board.visible_image_payload().evaluated_input
        assert board._interaction_is_pending()
        with pytest.raises(ValueError, match="pending image viewport revision"):
            board.present(
                _frame(
                    2,
                    viewport=_viewport(
                        revision=committed[0].viewport.viewport_revision,
                        bounds=(0.0, 0.0, 0.5, 0.5),
                    ),
                )
            )
        assert board.front_frame.sequence == 1
        assert board._interaction_is_pending()
        assert not board.discard_pending_image_interaction(
            replace(origin, sequence=origin.sequence + 1)
        )
        assert board._interaction_is_pending()
        assert board.discard_pending_image_interaction(origin)
        assert not board._interaction_is_pending()
        assert not board.discard_pending_image_interaction(origin)
    finally:
        board.close()
        application.processEvents()


def test_interaction_callback_fault_disables_and_releases_pending_state() -> None:
    def fail(_command) -> None:
        raise RuntimeError("display commit failed")

    application, board = _bound_board(_frame(1), interaction_callback=fail)
    try:
        _wheel(board, _point(_target(board), 0.5, 0.5), -120)
        assert board.selector_fault is not None
        assert "display commit failed" in str(board.selector_fault)
        assert not board._selector_enabled
        assert not board._interaction_is_pending()
        assert board._selector_hold is None
    finally:
        board.close()
        application.processEvents()


def test_viewport_crop_changes_painted_aspect_from_exact_visible_bounds() -> None:
    from PyQt5 import QtCore, QtGui

    cropped = _viewport(revision=4, bounds=(0.0, 0.0, 0.5, 1.0))
    application, board = _bound_board(_frame(1, viewport=cropped))
    try:
        target = _target(board)
        assert target.height() == 180
        assert target.width() == 90
        rendered = QtGui.QImage(board.size(), QtGui.QImage.Format_RGBA8888)
        rendered.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(rendered)
        try:
            board.render(painter)
        finally:
            painter.end()
        indexed = board._front[1][0][1]
        x = target.center().x()
        assert rendered.pixelColor(x, target.top() + target.height() // 4).rgba() == indexed.color(1)
        assert rendered.pixelColor(x, target.top() + 3 * target.height() // 4).rgba() == indexed.color(0)
    finally:
        board.close()
        application.processEvents()


def test_panel_revision_preserves_applied_and_cross_but_rejects_stale_front() -> None:
    from PyQt5 import QtCore, QtTest

    first = _frame(1)
    application, board = _bound_board(first, interaction_callback=lambda _value: None)
    try:
        viewport = first.panels[0].image_payload.viewport
        selection = viewport.selection_for_normalized_bounds((0.0, 0.0, 0.5, 0.5))
        board.set_selector_applied_selection(selection)
        target = _target(board)
        center = _point(target, 0.75, 0.75)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
        cross = board._cross_sample

        # An active draft is tied to revision 0 and must be cancelled when the
        # display-only panel revision changes.
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.1, 0.1),
        )
        replacement_viewport = _viewport(revision=1)
        replacement = _frame(2, viewport=replacement_viewport)
        board.present(replacement)
        assert board._selector_hold is None
        assert board._selector_draft_bounds is None
        assert board._selector_applied_bounds is not None
        assert board._cross_sample is cross
        assert board._selector_viewport == replacement_viewport

        with pytest.raises(ValueError, match="stale image viewport revision"):
            board.present(_frame(3, viewport=_viewport(revision=0)))
        assert board.front_frame is replacement

        # A real document identity change is structural and clears display
        # overlays, even though its new viewport revision may restart at zero.
        board.present(
            _frame(
                4,
                viewport=_viewport(revision=0),
                document_revision=1,
            )
        )
        assert board._selector_applied_bounds is None
        assert board._cross_sample is None
    finally:
        board.close()
        application.processEvents()
