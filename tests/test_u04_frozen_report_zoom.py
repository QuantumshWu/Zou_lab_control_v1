"""A frozen report page can be magnified.

The design allows exactly one frozen-raster exception - the multi-page
calibration report - and it attaches a condition: `报告类多页至少补 zoom`
(UX-003 and UX-006 both carry it).  `QtImageBoard` had no zoom at all, so a
report page could only ever be read at whatever size the window happened to be.
That is the one raster an operator cannot ask the system to re-render larger,
which is what makes the omission a real loss rather than a cosmetic one.

Zoom is opt-in on the presenter.  A LIVE board's zoom belongs to its
`ViewportTransform` at a matching revision (frozen clause S12.5); handing the
presenter a second, independent zoom would give a live panel two owners of the
same concern.  So the flag is off by default and the report window turns it on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import QtImageBoard


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return ensure_qt_app()


def _png_bytes(width: int = 40, height: int = 20) -> bytes:
    image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(10, 20, 30))
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    assert image.save(buffer, "PNG")
    return bytes(buffer.data())


def _board(app, *, zoomable: bool) -> QtImageBoard:
    board = QtImageBoard("page", zoomable=zoomable)
    board.resize(400, 200)
    board.present_encoded(_png_bytes())
    return board


def _wheel(board: QtImageBoard, steps: int, pos=None):
    point = pos if pos is not None else board.rect().center()
    board.wheelEvent(
        QtGui.QWheelEvent(
            QtCore.QPointF(point),
            QtCore.QPointF(board.mapToGlobal(point)),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 120 * steps),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.NoScrollPhase,
            False,
        )
    )


def test_a_report_page_magnifies_on_the_wheel(app):
    board = _board(app, zoomable=True)
    try:
        assert board.view_scale == 1.0
        _wheel(board, 1)
        assert board.view_scale > 1.0
        _wheel(board, 3)
        assert board.view_scale > 1.5
    finally:
        board.deleteLater()


def test_a_live_presenter_keeps_no_second_zoom_owner(app):
    """Off by default: a live panel's zoom belongs to its ViewportTransform."""

    board = _board(app, zoomable=False)
    try:
        _wheel(board, 4)
        assert board.view_scale == 1.0
        assert board.view_center == (0.5, 0.5)
    finally:
        board.deleteLater()


def test_zooming_out_stops_at_the_whole_page(app):
    board = _board(app, zoomable=True)
    try:
        _wheel(board, 3)
        _wheel(board, -20)
        assert board.view_scale == 1.0
        # Fully zoomed out the page is centred; there is no off-page margin.
        assert board.view_center == (0.5, 0.5)
    finally:
        board.deleteLater()


def test_magnification_is_bounded(app):
    board = _board(app, zoomable=True)
    try:
        _wheel(board, 200)
        assert board.view_scale == QtImageBoard.MAX_VIEW_SCALE
    finally:
        board.deleteLater()


def test_the_visible_window_never_leaves_the_page(app):
    """Whatever the operator does, no empty margin may appear."""

    board = _board(app, zoomable=True)
    try:
        rect = board.rect()
        for corner in (
            rect.topLeft() + QtCore.QPoint(2, 2),
            rect.topRight() + QtCore.QPoint(-2, 2),
            rect.bottomLeft() + QtCore.QPoint(2, -2),
            rect.bottomRight() + QtCore.QPoint(-2, -2),
        ):
            _wheel(board, 6, corner)
            cx, cy = board.view_center
            half = 0.5 / board.view_scale
            assert cx - half >= -1e-9 and cx + half <= 1 + 1e-9
            assert cy - half >= -1e-9 and cy + half <= 1 + 1e-9
    finally:
        board.deleteLater()


def test_dragging_pans_only_while_magnified(app):
    board = _board(app, zoomable=True)
    try:
        start = board.rect().center()

        # Un-magnified, a drag must not move anything.
        board.mousePressEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(start),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert board._pan_from is None

        _wheel(board, 6)
        before = board.view_center
        board.mousePressEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(start),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        board.mouseMoveEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove,
                QtCore.QPointF(start + QtCore.QPoint(30, 0)),
                QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        board.mouseReleaseEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonRelease,
                QtCore.QPointF(start + QtCore.QPoint(30, 0)),
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert board.view_center != before
        assert board._pan_from is None
    finally:
        board.deleteLater()


def test_double_click_returns_to_the_whole_page_only_when_magnified(app):
    board = _board(app, zoomable=True)
    seen = []
    board.normalizedDoubleClicked.connect(lambda x, y: seen.append((x, y)))
    try:
        _wheel(board, 6)
        assert board.view_scale > 1.0
        board.mouseDoubleClickEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QPointF(board.rect().center()),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert board.view_scale == 1.0
        assert seen == [], "a reset must not also be reported as a pick"

        # Un-magnified the gesture keeps the meaning existing consumers rely on.
        board.mouseDoubleClickEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QPointF(board.rect().center()),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        assert len(seen) == 1
    finally:
        board.deleteLater()


def test_a_same_size_refresh_keeps_the_operators_zoom(app):
    board = _board(app, zoomable=True)
    try:
        _wheel(board, 6)
        kept = board.view_scale
        board.present_encoded(_png_bytes())
        assert board.view_scale == kept, "a redrawn page is the same picture"

        board.present_encoded(_png_bytes(80, 40))
        assert board.view_scale == 1.0, "a different raster starts over"
    finally:
        board.deleteLater()


def test_the_report_window_turns_zoom_on():
    source = (
        ROOT / "Zou_lab_control" / "workbench" / "_frozen_raster.py"
    ).read_text(encoding="utf-8")
    assert "zoomable=True" in source
