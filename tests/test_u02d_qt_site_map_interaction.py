"""Qt oracle for one exact physical SiteMap composite front."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_qt_and_agg_consume_one_painter_neutral_site_ring_vocabulary():
    from zlc_frontend.site_map import (
        SITE_EMPTY_ALPHA,
        SITE_EMPTY_COLOR,
        SITE_EMPTY_LINEWIDTH,
        SITE_INVALID_COLOR,
        SITE_OCCUPIED_ALPHA,
        SITE_OCCUPIED_COLOR,
        SITE_OCCUPIED_LINEWIDTH,
    )
    from zlc_frontend.render_style import FIT_FAILURE_COLOR, SITE_OCCUPANCY_STYLE

    assert (SITE_EMPTY_COLOR, SITE_EMPTY_ALPHA, SITE_EMPTY_LINEWIDTH) == (
        SITE_OCCUPANCY_STYLE["empty"]["color"],
        SITE_OCCUPANCY_STYLE["empty"]["alpha"],
        SITE_OCCUPANCY_STYLE["empty"]["linewidth"],
    )
    assert (SITE_OCCUPIED_COLOR, SITE_OCCUPIED_ALPHA, SITE_OCCUPIED_LINEWIDTH) == (
        SITE_OCCUPANCY_STYLE["occupied"]["color"],
        SITE_OCCUPANCY_STYLE["occupied"]["alpha"],
        SITE_OCCUPANCY_STYLE["occupied"]["linewidth"],
    )
    assert SITE_INVALID_COLOR == FIT_FAILURE_COLOR


def _evaluated_input(name: str, revision: int):
    from zlc_data import (
        BlockId,
        DatasetRevision,
        DatasetRevisionRef,
        StreamGenerationId,
    )
    from zlc_frontend.figure import DatasetId, EvaluatedInput

    schema = ("a" if name == "occupancy" else "b") * 64
    return EvaluatedInput(
        DatasetId(name),
        DatasetRevisionRef(
            BlockId(f"{name}-block"),
            StreamGenerationId(f"{name}-generation"),
            schema,
            DatasetRevision(revision),
        ),
    )


def _frame(
    sequence: int,
    *,
    payload_from=None,
    x_step: float = 1.0,
    y_step: float = 1.0,
):
    from zlc_data import (
        AxisId,
        AxisSpec,
        CoordinateFrameId,
        SITE,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.display_range import RelimMode
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedImage
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.image_raster import rasterize_image_indexed8
    from zlc_frontend.image_view import ImageViewportTransform
    from zlc_frontend.render import (
        BoardFrame,
        CoherenceStamp,
        ImagePanelPayload,
        PanelFrame,
        PanelPresentationIdentity,
        SiteMapPanelPayload,
        SourceIdentity,
    )

    size = 64
    coordinate_frame = CoordinateFrameId("qcmos.roi0.bin1")
    x_axis = AxisSpec(
        AxisId("sites.frame-x"),
        "camera x",
        SPATIAL_X,
        size,
        tuple(float(index) * x_step for index in range(size)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    y_axis = AxisSpec(
        AxisId("sites.frame-y"),
        "camera y",
        SPATIAL_Y,
        size,
        tuple(float(index) * y_step for index in range(size)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    viewport = ImageViewportTransform((y_axis, x_axis), viewport_revision=3)
    values = np.arange(size * size, dtype=np.float64).reshape(size, size)
    evaluated = EvaluatedImage(
        EvaluatedAxis(
            x_axis.axis_id,
            x_axis.name,
            x_axis.role,
            x_axis.unit,
            tuple(range(size)),
            x_axis.coordinates,
        ),
        EvaluatedAxis(
            y_axis.axis_id,
            y_axis.name,
            y_axis.role,
            y_axis.unit,
            tuple(range(size)),
            y_axis.coordinates,
        ),
        values,
        np.ones(values.shape, dtype=bool),
    )
    background_input = _evaluated_input("background", sequence + 1)
    raster, data_range, histogram, limits = rasterize_image_indexed8(
        evaluated,
        ImageDisplayState(
            revision=3,
            relim_mode=RelimMode.FIXED,
            fixed_color_limits=(0.0, float(values.max())),
        ),
        current_color_limits=None,
        previous_relim_mode=None,
    )
    background = ImagePanelPayload(
        evaluated,
        background_input,
        viewport,
        data_range,
        histogram,
        tuple(0xFF000000 | index * 0x00010101 for index in range(256)),
        limits,
    )
    occupancy_input = _evaluated_input("occupancy", sequence + 1)
    centers = np.asarray(((16.0, 16.0), (32.0, 32.0), (48.0, 48.0)))
    payload = SiteMapPanelPayload(
        background,
        occupancy_input,
        AxisSpec(AxisId("sites.site"), "site", SITE, 3, ("A", "B", "C")),
        coordinate_frame,
        centers,
        np.asarray((False, True, False), dtype=bool),
        np.asarray((True, True, False), dtype=bool),
        "calibration:exact-map-v1",
        f"repeat=0;point={sequence}",
    )
    if payload_from is not None:
        payload = payload_from(payload)
    presentation = PanelPresentationIdentity(
        "sites",
        "occupancy-cell-document",
        1,
        0,
        payload.background.viewport.viewport_revision,
    )
    stamp = CoherenceStamp(
        "occupancy-final",
        "occupancy-final-epoch",
        "exact-occupancy-camera-cell",
        "c" * 64,
        payload.join_key_digest,
        (payload.occupancy_input, payload.background.evaluated_input),
        (presentation,),
    )
    source = SourceIdentity(
        payload.occupancy_input.dataset_id,
        payload.occupancy_input.ref.block_id,
        payload.occupancy_input.ref.stream_generation,
        payload.occupancy_input.ref.schema_fingerprint,
    )
    return BoardFrame(
        "site-map-board",
        0,
        sequence,
        (PanelFrame("sites", "occupancy-cell", source, stamp, raster, payload),),
    )


def _point(target, x: float, y: float):
    from PyQt5 import QtCore

    return QtCore.QPoint(
        target.left() + round(x * (target.width() - 1)),
        target.top() + round(y * (target.height() - 1)),
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


def test_site_map_uses_background_lut_and_paints_all_three_site_states():
    from PyQt5 import QtCore, QtGui
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    frame = _frame(0)
    payload = frame.panels[0].display_payload
    board = QtRasterBoard(("sites",), columns=1)
    board.resize(320, 260)
    board.show()
    try:
        board.present(frame)
        board.bind_rectangle_selector(
            "sites",
            payload.background.viewport,
            lambda _gesture: None,
            enabled=True,
            interaction_callback=lambda _command: None,
        )
        application.processEvents()
        assert board.visible_site_map_payload() is payload
        assert board.visible_image_payload() is payload.background
        assert tuple(board._front[1][0][1].colorTable()) == payload.background.base_palette

        target = board._selector_target()[0]
        canvas = QtGui.QImage(board.size(), QtGui.QImage.Format_ARGB32)
        canvas.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(canvas)
        board._paint_site_map_rings(painter, payload, target)
        painter.end()
        for full_point in payload.full_normalized_centers_xy:
            normalized = payload.background.viewport.visible_point_for_full_point(
                tuple(full_point)
            )
            center = _point(target, *normalized)
            colors = {
                canvas.pixelColor(x, y).name().upper()
                for x in range(max(0, center.x() - 32), min(canvas.width(), center.x() + 33))
                for y in range(max(0, center.y() - 32), min(canvas.height(), center.y() + 33))
                if canvas.pixelColor(x, y).alpha() > 0
            }
            assert colors
        all_colors = {
            canvas.pixelColor(x, y).name().upper()
            for x in range(canvas.width())
            for y in range(canvas.height())
            if canvas.pixelColor(x, y).alpha() > 0
        }
        assert {"#FFFFFF", "#D07850", "#CD7380"} <= all_colors
    finally:
        board.close()


def test_site_map_uses_physical_axis_aspect_and_west_anchor_for_anisotropic_pixels():
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    frame = _frame(20, x_step=2.0, y_step=4.0)
    payload = frame.panels[0].display_payload
    board = QtRasterBoard(("sites",), columns=1)
    board.resize(620, 300)
    board.show()
    try:
        board.present(frame)
        board.bind_rectangle_selector(
            "sites",
            payload.background.viewport,
            lambda _gesture: None,
            enabled=True,
            interaction_callback=lambda _command: None,
        )
        application.processEvents()
        target = board._selector_target()[0]
        assert target.left() == board.rect().left()
        assert target.width() / target.height() == pytest.approx(0.5, abs=0.01)
        ring_width, ring_height = payload.visible_ring_span
        assert ring_width * target.width() == pytest.approx(
            ring_height * target.height(),
            rel=0.02,
        )
    finally:
        board.close()


def test_site_map_reuses_image_gestures_but_keeps_area_display_only():
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.selector import ImageViewportCommit
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    frame = _frame(1)
    payload = frame.panels[0].display_payload
    gestures = []
    commands = []
    board = QtRasterBoard(("sites",), columns=1)
    board.resize(320, 260)
    board.show()
    try:
        board.present(frame)
        def accept_display_only(gesture):
            gestures.append(gesture)
            with pytest.raises(RuntimeError, match="display-only"):
                board.selection_for_rectangle_gesture(gesture)

        board.bind_rectangle_selector(
            "sites",
            payload.background.viewport,
            accept_display_only,
            enabled=True,
            interaction_callback=commands.append,
        )
        application.processEvents()
        target = board._selector_target()[0]
        start = _point(target, 0.2, 0.2)
        end = _point(target, 0.7, 0.7)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.mouseMove(board, end)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
        assert len(gestures) == 1
        gesture = gestures[0]
        board.set_image_rectangle_candidate(gesture.normalized_bounds)
        assert board._selector_applied_bounds == gesture.normalized_bounds
        assert board.visible_image_origin().evaluated_input == payload.occupancy_input

        _wheel(board, _point(target, 0.5, 0.5), -120)
        assert len(commands) == 1
        assert isinstance(commands[0], ImageViewportCommit)
        assert commands[0].origin.evaluated_input == payload.occupancy_input
        assert commands[0].viewport.axes == payload.background.viewport.axes
    finally:
        board.close()


def test_site_map_hold_survives_same_selection_data_revision_but_not_geometry_change():
    from PyQt5 import QtCore, QtTest
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()
    first = _frame(2)
    payload = first.panels[0].display_payload
    board = QtRasterBoard(("sites",), columns=1)
    board.resize(320, 260)
    board.show()
    try:
        board.present(first)
        board.bind_rectangle_selector(
            "sites",
            payload.background.viewport,
            lambda _gesture: None,
            enabled=True,
            interaction_callback=lambda _command: None,
        )
        application.processEvents()
        target = board._selector_target()[0]
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target, 0.2, 0.2),
        )
        assert board._selector_hold is not None
        board.present(_frame(3))
        assert board._selector_hold is not None

        changed = _frame(
            4,
            payload_from=lambda value: replace(
                value,
                calibration_identity="calibration:exact-map-v2",
            ),
        )
        board.present(changed)
        assert board._selector_hold is None
    finally:
        board.close()


def test_site_map_hold_rejects_every_background_producer_identity_change():
    from PyQt5 import QtCore, QtTest
    from zlc_data import BlockId, StreamGenerationId
    from zlc_frontend.figure import DatasetId
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app

    application = ensure_qt_app()

    def changed_background(payload, kind: str):
        original = payload.background.evaluated_input
        reference = original.ref
        if kind == "dataset":
            changed_input = replace(
                original,
                dataset_id=DatasetId("background-other"),
            )
        elif kind == "block":
            changed_input = replace(
                original,
                ref=replace(reference, block_id=BlockId("background-block-other")),
            )
        elif kind == "generation":
            changed_input = replace(
                original,
                ref=replace(
                    reference,
                    stream_generation=StreamGenerationId("background-generation-other"),
                ),
            )
        else:
            changed_input = replace(
                original,
                ref=replace(reference, schema_fingerprint="d" * 64),
            )
        return replace(
            payload,
            background=replace(
                payload.background,
                evaluated_input=changed_input,
            ),
        )

    for offset, kind in enumerate(("dataset", "block", "generation", "schema")):
        first = _frame(10 + offset * 2)
        payload = first.panels[0].display_payload
        board = QtRasterBoard(("sites",), columns=1)
        board.resize(320, 260)
        board.show()
        try:
            board.present(first)
            board.bind_rectangle_selector(
                "sites",
                payload.background.viewport,
                lambda _gesture: None,
                enabled=True,
                interaction_callback=lambda _command: None,
            )
            application.processEvents()
            target = board._selector_target()[0]
            QtTest.QTest.mousePress(
                board,
                QtCore.Qt.LeftButton,
                pos=_point(target, 0.2, 0.2),
            )
            assert board._selector_hold is not None
            board.present(
                _frame(
                    11 + offset * 2,
                    payload_from=lambda value, selected=kind: changed_background(
                        value,
                        selected,
                    ),
                )
            )
            assert board._selector_hold is None
        finally:
            board.close()
