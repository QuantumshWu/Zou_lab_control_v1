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
    panel_id: str = "image",
    board_id: str = "camera-board",
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
    from zlc_frontend.display_range import RelimMode
    from zlc_frontend.image_display import ImageDisplayState
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
            viewport.x_axis.role,
            viewport.x_axis.unit,
            tuple(range(viewport.x_axis.size)),
            tuple(viewport.x_axis.coordinates),
        ),
        EvaluatedAxis(
            viewport.y_axis.axis_id,
            viewport.y_axis.name,
            viewport.y_axis.role,
            viewport.y_axis.unit,
            tuple(range(viewport.y_axis.size)),
            tuple(viewport.y_axis.coordinates),
        ),
        values,
        np.ones(values.shape, dtype=bool),
    )
    display = ImageDisplayState(
        revision=viewport.viewport_revision,
        relim_mode=RelimMode.FIXED,
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
        panel_id,
        f"{panel_id}-document",
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
        board_id,
        0,
        sequence,
        (PanelFrame(panel_id, "camera", source, stamp, raster, payload),),
    )


def _binding(board, panel_id: str = "image"):
    return board._image_bindings[panel_id]


def _target(board, panel_id: str = "image"):
    return board._selector_target(_binding(board, panel_id))[0]


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
        frame.panels[0].display_payload.viewport,
        (lambda _gesture: None) if selection_callback is None else selection_callback,
        enabled=True,
        interaction_callback=interaction_callback,
    )
    application.processEvents()
    return application, board


def _two_image_frame(sequence: int, *, second_offset: float = 0.0):
    from zlc_frontend.render import BoardFrame

    first = replace(
        _frame(
            sequence,
            panel_id="image-a",
            board_id="two-image-board",
            source_suffix="-a",
            values=[[1.0, 2.0], [3.0, 4.0]],
        ).panels[0],
        coherence_group="image-a",
    )
    second = replace(
        _frame(
            sequence,
            panel_id="image-b",
            board_id="two-image-board",
            source_suffix="-b",
            values=np.asarray([[10.0, 20.0], [30.0, 40.0]]) + second_offset,
            color_limits=(0.0, 100.0),
        ).panels[0],
        coherence_group="image-b",
    )
    return BoardFrame(
        "two-image-board",
        0,
        sequence,
        (first, second),
    )


def _with_radial_fit_overlay(
    panel,
    *,
    status,
    caption: str,
    center=None,
    radius=None,
    diagnostic: str = "",
):
    from zlc_frontend.render import RadialGaussianImageFitOverlay

    payload = panel.display_payload
    overlay = RadialGaussianImageFitOverlay(
        payload.evaluated_input.ref,
        "saved-radial-fit-v1",
        None if status is None else 0,
        status,
        payload.viewport.coordinate_frame,
        caption,
        diagnostic,
        center,
        radius,
    )
    return replace(panel, display_payload=replace(payload, fit_overlay=overlay))


def test_radial_fit_overlay_geometry_uses_exact_view_without_clamping() -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_data import FitBatchStatus
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    _application = ensure_qt_app()
    panel = _with_radial_fit_overlay(
        _frame(1).panels[0],
        status=FitBatchStatus.CONVERGED,
        caption="repeat=0",
        center=(10.5, 25.0),
        radius=0.25,
    )
    target = QtCore.QRect(10, 20, 200, 100)
    center, ring = QtRasterBoard._radial_fit_overlay_geometry(
        panel.display_payload,
        target,
    )
    assert (center.x(), center.y()) == pytest.approx((110.0, 70.0))
    assert (ring.x(), ring.y(), ring.width(), ring.height()) == pytest.approx(
        (85.0, 68.75, 50.0, 2.5)
    )

    zoomed = _viewport(bounds=(0.25, 0.25, 0.75, 0.75))
    off_view = _with_radial_fit_overlay(
        _frame(2, viewport=zoomed).panels[0],
        status=FitBatchStatus.CONVERGED,
        caption="repeat=1",
        center=(8.0, 10.0),
        radius=0.1,
    ).display_payload
    off_center, off_ring = QtRasterBoard._radial_fit_overlay_geometry(
        off_view,
        target,
    )
    assert off_center.x() < target.left()
    assert off_center.y() < target.top()
    assert off_ring.right() < target.left()
    assert off_ring.bottom() < target.top()

    canvas = QtGui.QImage(target.size(), QtGui.QImage.Format_ARGB32)
    canvas.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(canvas)
    try:
        QtRasterBoard._paint_radial_fit_overlay(
            painter,
            off_view,
            QtCore.QRect(QtCore.QPoint(), target.size()),
        )
    finally:
        painter.end()
    # Only the compact top caption is visible; the off-view geometry was not
    # clamped into the image body.
    assert any(
        canvas.pixelColor(x, y).alpha() > 0
        for x in range(canvas.width())
        for y in range(min(32, canvas.height()))
    )
    assert not any(
        canvas.pixelColor(x, y).alpha() > 0
        for x in range(canvas.width())
        for y in range(40, canvas.height())
    )


def test_radial_fit_board_target_preserves_physical_circle_aspect() -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_data import FitBatchStatus
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app
    from zlc_frontend.qt_widgets._raster_front import _panel_image_geometry

    _application = ensure_qt_app()
    payload = _with_radial_fit_overlay(
        _frame(1).panels[0],
        status=FitBatchStatus.CONVERGED,
        caption="repeat=0",
        center=(10.5, 25.0),
        radius=0.25,
    ).display_payload
    image = QtGui.QImage(2, 2, QtGui.QImage.Format_Indexed8)
    target, _source, _rail = _panel_image_geometry(
        QtCore.QRect(0, 0, 240, 200),
        image,
        payload,
    )
    _center, ring = QtRasterBoard._radial_fit_overlay_geometry(payload, target)

    # X cells are 1 unit wide while Y cells are 10 units high.  The image
    # body therefore becomes ten times taller than wide, and the physical
    # radial contour remains circular on screen to integer-pixel tolerance.
    assert target.height() / target.width() == pytest.approx(10.0, rel=0.06)
    assert ring.width() == pytest.approx(ring.height(), abs=1.0)


@pytest.mark.parametrize(
    ("status", "diagnostic", "expected"),
    (
        (None, "NOT_PRESENT", "NOT_PRESENT"),
        ("NO_VALID_DATA", "NO_VALID_DATA: dead site", "NO_VALID_DATA"),
    ),
)
def test_sparse_and_failed_fit_cells_paint_status_but_no_geometry(
    status,
    diagnostic: str,
    expected: str,
) -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_data import FitBatchStatus
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    _application = ensure_qt_app()
    typed_status = None if status is None else FitBatchStatus(status)
    payload = _with_radial_fit_overlay(
        _frame(1).panels[0],
        status=typed_status,
        caption="site=(2, 3)",
        diagnostic=diagnostic,
    ).display_payload
    assert (
        QtRasterBoard._radial_fit_overlay_geometry(
            payload,
            QtCore.QRect(0, 0, 220, 120),
        )
        is None
    )
    assert QtRasterBoard._radial_fit_caption_status(payload.fit_overlay) == (
        f"site=(2, 3) · {expected}"
    )

    canvas = QtGui.QImage(220, 120, QtGui.QImage.Format_ARGB32)
    canvas.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(canvas)
    try:
        QtRasterBoard._paint_radial_fit_overlay(
            painter,
            payload,
            QtCore.QRect(0, 0, 220, 120),
        )
    finally:
        painter.end()
    assert any(
        canvas.pixelColor(x, y).alpha() > 0
        for x in range(canvas.width())
        for y in range(32)
    )
    assert not any(
        canvas.pixelColor(x, y).alpha() > 0
        for x in range(canvas.width())
        for y in range(40, canvas.height())
    )


def test_two_image_front_paints_fit_vectors_without_mutating_indexed8() -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_data import FitBatchStatus
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app
    from zlc_frontend.qt_widgets.style import ORANGE

    application = ensure_qt_app()
    base = _two_image_frame(1)
    frame = replace(
        base,
        panels=(
            _with_radial_fit_overlay(
                base.panels[0],
                status=FitBatchStatus.CONVERGED,
                caption="repeat=0",
                center=(10.5, 25.0),
                radius=0.25,
            ),
            _with_radial_fit_overlay(
                base.panels[1],
                status=None,
                caption="repeat=1",
                diagnostic="NOT_PRESENT",
            ),
        ),
    )
    board = QtRasterBoard(("image-a", "image-b"), columns=2)
    board.resize(480, 180)
    board.show()
    try:
        board.present(frame)
        for panel in frame.panels:
            board.bind_rectangle_selector(
                panel.panel_id,
                panel.display_payload.viewport,
                lambda _gesture: None,
            )
        application.processEvents()
        for index, panel in enumerate(frame.panels):
            owner, prepared = board._front[1][index]
            assert owner is panel.raster.pixels
            assert prepared.format() == QtGui.QImage.Format_Indexed8
            assert bytes(owner) == panel.raster.pixels
            assert board.visible_image_payload(panel.panel_id) is panel.display_payload

        canvas = QtGui.QImage(board.size(), QtGui.QImage.Format_ARGB32)
        canvas.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(canvas)
        try:
            board.render(painter)
        finally:
            painter.end()

        orange = QtGui.QColor(ORANGE).name().upper()
        targets = (_target(board, "image-a"), _target(board, "image-b"))

        def body_orange(target):
            return sum(
                canvas.pixelColor(x, y).name().upper() == orange
                for x in range(target.left(), target.right() + 1)
                for y in range(target.top() + 40, target.bottom() + 1)
            )

        assert body_orange(targets[0]) > 0
        assert body_orange(targets[1]) == 0
    finally:
        board.close()
        application.processEvents()


def test_two_image_bindings_isolate_front_hold_state_fault_and_left_double_click() -> None:
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    first = _two_image_frame(1)
    gestures = {"image-a": [], "image-b": []}
    image_commands = {"image-a": [], "image-b": []}
    board = QtRasterBoard(("image-a", "image-b"), columns=2)
    board.resize(480, 180)
    board.show()
    board.present(first)
    for panel in first.panels:
        board.bind_rectangle_selector(
            panel.panel_id,
            panel.display_payload.viewport,
            gestures[panel.panel_id].append,
            interaction_callback=image_commands[panel.panel_id].append,
        )
    application.processEvents()
    try:
        a = _binding(board, "image-a")
        b = _binding(board, "image-b")
        target_a = _target(board, "image-a")
        target_b = _target(board, "image-b")

        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=_point(target_b, 0.75, 0.75),
        )
        assert b.cross is not None and a.cross is None
        board.set_image_interaction_readiness("image-b", False)
        assert board._image_interaction_armed(a)
        assert not board._image_interaction_armed(b)
        board.set_image_interaction_readiness("image-b", True)
        _drag_move(board, _point(target_b, 0.25, 0.25), QtCore.Qt.NoButton)
        assert b.hover is not None and a.hover is None
        _drag_move(board, _point(target_a, 0.25, 0.25), QtCore.Qt.NoButton)
        assert a.hover is not None and b.hover is None

        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target_a, 0.1, 0.1),
        )
        _drag_move(board, _point(target_a, 0.8, 0.8), QtCore.Qt.LeftButton)
        assert board._selector_hold is not None
        assert board._selector_hold.panel_id == "image-a"
        held_a = board.visible_image_payload("image-a")
        old_b = board.visible_image_payload("image-b")

        board.present(_two_image_frame(2, second_offset=100.0))
        assert board.front_frame.sequence == 2
        assert board.visible_image_payload("image-a") is held_a
        assert board.visible_image_payload("image-b") is not old_b
        assert board.visible_image_payload("image-b").image.values[1, 1] == 140.0
        assert b.cross is not None

        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target_a, 0.8, 0.8),
        )
        assert len(gestures["image-a"]) == 1
        assert gestures["image-b"] == []
        assert board._selector_hold is None

        assert _wheel(
            board,
            _point(_target(board, "image-b"), 0.5, 0.5),
            -120,
        ).isAccepted()
        b_command = image_commands["image-b"][-1]
        assert b.pending_viewport is not None
        assert a.pending_viewport is None and a.pending_color_limits is None
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=_point(_target(board, "image-a"), 0.25, 0.25),
        )
        assert a.cross is not None and b.cross is not None
        assert board.discard_pending_image_interaction(b_command.origin)
        assert b.pending_viewport is None

        area = a.draft_bounds
        assert area is not None
        board.set_image_rectangle_candidate(area, panel_id="image-a")
        assert b.applied_bounds is None
        board.set_selector_draft_selection(None, panel_id="image-a")
        visible_a = board.visible_image_payload("image-a")
        assert visible_a is not None
        a_viewport = visible_a.viewport
        a_color_limits = visible_a.color_limits
        a_cross = a.cross
        double_clicked = []
        board.imagePanelLeftDoubleClicked.connect(double_clicked.append)
        QtTest.QTest.mouseDClick(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(_target(board, "image-a"), 0.5, 0.5),
        )
        assert double_clicked == ["image-a"]
        assert a.applied_bounds == area
        assert a.draft_bounds is None
        assert a.pending_viewport is None and a.pending_color_limits is None
        assert board.visible_image_payload("image-a") is visible_a
        assert visible_a.viewport == a_viewport
        assert visible_a.color_limits == a_color_limits
        assert a.cross is a_cross

        def fail(_command) -> None:
            raise RuntimeError("image-b display failure")

        board.bind_rectangle_selector(
            "image-b",
            b.viewport,
            gestures["image-b"].append,
            interaction_callback=fail,
        )
        assert _wheel(board, _point(_target(board, "image-b"), 0.5, 0.5), -120).isAccepted()
        assert board.image_selector_fault("image-b") is not None
        assert board.image_selector_fault("image-a") is None
        assert a.binding_enabled
    finally:
        board.close()
        application.processEvents()


def test_indexed_lut_and_exact_cross_never_infer_z_from_display_code() -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    frame = _frame(1)
    application, board = _bound_board(frame)
    try:
        payload = frame.panels[0].display_payload
        assert payload is not None
        qimage = board._front[1][0][1]
        assert tuple(qimage.colorTable()) == payload.base_palette

        position = _point(_target(board), 0.75, 0.25)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=position)
        sample = _binding(board).cross
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
        invalid = _binding(board).cross
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
        panels=(replace(frame.panels[0], display_payload=None),),
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
                display_payload=None,
            ),
        ),
    )
    viewport = exact.panels[0].display_payload.viewport
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
        def hover_at(point) -> None:
            board.mouseMoveEvent(
                QtGui.QMouseEvent(
                    QtCore.QEvent.MouseMove,
                    QtCore.QPointF(point),
                    QtCore.Qt.NoButton,
                    QtCore.Qt.NoButton,
                    QtCore.Qt.NoModifier,
                )
            )

        position = _point(_target(board), 0.75, 0.75)
        hover_at(position)
        sample = _binding(board).hover
        assert sample is not None and sample.value == 4000.0 and sample.valid

        # A stationary pointer remains useful on a live image: every promoted
        # front re-evaluates it against the new exact payload/provenance.
        board.present(_frame(2, values=[[1.0, 2.0], [3.0, 8.0]]))
        application.processEvents()
        sample = _binding(board).hover
        assert sample is not None and sample.value == 8.0 and sample.valid

        QtWidgets.QApplication.sendEvent(board, QtCore.QEvent(QtCore.QEvent.Leave))
        assert _binding(board).hover is None
        hover_at(position)
        assert _binding(board).hover is not None
        board.set_selectors_enabled(False)
        assert _binding(board).hover is None
        assert not _wheel(board, position, -120).isAccepted()

        board.set_selectors_enabled(True)
        hover_at(position)
        assert _binding(board).hover is not None
        QtWidgets.QApplication.sendEvent(board, QtGui.QHideEvent())
        assert _binding(board).hover is None
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
        bounds = _binding(board).draft_bounds
        assert bounds is not None
        label = board._selection_endpoint_label(_binding(board), bounds)
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
        selection = board._require_selector_viewport(
            _binding(board)
        ).selection_for_normalized_bounds(bounds)
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
        label = board._selection_endpoint_label(
            _binding(board),
            (0.0, 0.0, 1.0, 1.0),
        )
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
        label = board._selection_endpoint_label(_binding(board), bounds)
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
    payload = frame.panels[0].display_payload
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
        anchor = _binding(board).drag_anchor
        assert hold is not None and anchor is not None

        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        _mouse_press(board, _point(target, 0.5, 0.5), QtCore.Qt.MiddleButton)
        _mouse_press(board, _point(target, 0.5, 0.5), QtCore.Qt.RightButton)
        assert commands == [] and _binding(board).cross is None
        assert board._selector_hold is hold
        assert _binding(board).drag_anchor == anchor
        assert _binding(board).pan_anchor is None and _binding(board).clim_drag is None

        _drag_move(board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert len(gestures) == 1 and commands == []

        target = _target(board)
        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        assert len(commands) == 1 and board._image_interaction_is_pending(
            _binding(board)
        )
        # The pending IMAGE CAS is local to the image/rail. Black margin (or a
        # sibling cell in a larger board) must still bubble to its parent.
        assert not _wheel(board, QtCore.QPoint(1, 1), -120).isAccepted()
        assert _wheel(board, _point(target, 0.5, 0.5), -120).isAccepted()
        assert len(commands) == 1
    finally:
        board.close()
        application.processEvents()


def test_middle_pan_uses_press_pixels_holds_only_exact_payload_and_commits_live() -> None:
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
        assert hold.display_payload is first.panels[0].display_payload
        assert not isinstance(hold.display_payload, type(first))
        held_origin = board.visible_image_origin()
        assert held_origin is not None and held_origin.sequence == 1
        assert held_origin.input_identity is first.panels[0].display_payload.evaluated_input

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
        # The reference pans LIVE: the commit lands on the motion itself.
        assert len(committed) == 1
        expected = initial_viewport.panned_by_pixels(
            (24.0, 9.0),
            (target.width(), target.height()),
        )
        assert committed[0].viewport == expected
        QtTest.QTest.mouseRelease(board, QtCore.Qt.MiddleButton, pos=end)

        # Release only ends the gesture; the last candidate is not re-issued.
        assert len(committed) == 1
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
        viewport = first.panels[0].display_payload.viewport
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
        assert _binding(board).cross is not None
        QtTest.QTest.mouseDClick(board, QtCore.Qt.RightButton, pos=center)
        assert _binding(board).cross is None
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
        rail, *_rest, payload = board._clim_rail_target(_binding(board))
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
        assert board._selector_hold.display_payload is payload
        _drag_move(board, end, QtCore.Qt.LeftButton)
        # The reference's DragHLine recolors on EVERY motion: the commit goes
        # out live, while the painted LUT still waits for the owner's answer
        # (never a temporary LUT).
        assert len(committed) == 1
        assert tuple(board._front[1][0][1].colorTable()) == original_lut
        candidate_label = board._clim_candidate_label(_binding(board), payload)
        assert candidate_label.startswith("H low=")
        assert "high=3500" in candidate_label
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)

        # Release only ends the gesture; the last candidate is not re-issued.
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
        assert not board._image_interaction_is_pending(_binding(board))
        assert tuple(board._front[1][0][1].colorTable()) == original_lut
        assert board.front_frame.panels[0].raster.pixels != original_pixels

        # Every non-release lifecycle exit uses the same hold cleanup.
        rail, *_rest, payload = board._clim_rail_target(_binding(board))
        domain = board._color_rail_domain(payload)
        start = QtCore.QPoint(
            rail.center().x(),
            int(round(board._rail_y(payload.color_limits[0], domain, rail))),
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        assert board._selector_hold is not None
        board.set_selectors_enabled(False)
        assert board._selector_hold is None and _binding(board).clim_drag is None
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
        sample = _binding(board).cross
        assert sample is not None
        viewport = ImageViewportTransform(
            first.panels[0].display_payload.viewport.axes,
            1,
            (0.0, 0.0, 0.5, 0.5),
        )
        board.present(_frame(2, viewport=viewport))
        assert _binding(board).cross is sample

        painter = Mock()
        painter.fontMetrics.return_value = board.fontMetrics()
        board._paint_cross_sample(
            painter,
            _binding(board),
            sample,
            _target(board),
        )
        painter.drawLine.assert_not_called()
        painter.drawEllipse.assert_not_called()
        painter.fillRect.assert_called_once()
        assert "off-view" in painter.drawText.call_args.args[-1]

        QtTest.QTest.mouseDClick(
            board,
            QtCore.Qt.RightButton,
            pos=_point(_target(board), 0.5, 0.5),
        )
        assert _binding(board).cross is None
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
        assert origin.input_identity is board.visible_image_payload().evaluated_input
        assert board._image_interaction_is_pending(_binding(board))
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
        assert board._image_interaction_is_pending(_binding(board))
        assert not board.discard_pending_image_interaction(
            replace(origin, sequence=origin.sequence + 1)
        )
        assert board._image_interaction_is_pending(_binding(board))
        assert board.discard_pending_image_interaction(origin)
        assert not board._image_interaction_is_pending(_binding(board))
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
        assert board._selector_enabled
        assert not _binding(board).binding_enabled
        assert not board._image_interaction_is_pending(_binding(board))
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
        viewport = first.panels[0].display_payload.viewport
        selection = viewport.selection_for_normalized_bounds((0.0, 0.0, 0.5, 0.5))
        board.set_selector_applied_selection(selection)
        target = _target(board)
        center = _point(target, 0.75, 0.75)
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
        cross = _binding(board).cross

        # A display-only revision advance does NOT kill a live gesture: the
        # hold keeps the press frame frozen while the new front lands
        # underneath (the design's hold semantics).  Escape then releases the
        # gesture without touching the applied state.
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.1, 0.1),
        )
        replacement_viewport = _viewport(revision=1)
        replacement = _frame(2, viewport=replacement_viewport)
        board.present(replacement)
        assert board._selector_hold is not None
        QtTest.QTest.keyClick(board, QtCore.Qt.Key_Escape)
        assert board._selector_hold is None
        assert _binding(board).draft_bounds is None
        assert _binding(board).applied_bounds is not None
        assert _binding(board).cross is cross
        assert _binding(board).viewport == replacement_viewport

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
        assert _binding(board).applied_bounds is None
        assert _binding(board).cross is None
    finally:
        board.close()
        application.processEvents()
