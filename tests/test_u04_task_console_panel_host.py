"""A TaskConsole panel must answer gestures made on its visible board.

This is an operator-path test: it opens ``Experiment.task_console()``, adds both
panels through the header controls, arms Selectors through the visible switch,
then uses the mouse/wheel on the same raster boards the operator sees.  The
fixture payloads only stand in for monitor frames that have already arrived.
"""

from __future__ import annotations

from dataclasses import replace
import os

import pytest

import Zou_lab_control.notebook as zlc


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _select_combo_item(combo, value, application) -> None:
    """Choose one visible combo entry through its open popup keyboard path."""

    from PyQt5 import QtCore, QtTest

    index = combo.findData(value)
    assert index >= 0
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents()
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Home)
    for _ in range(index):
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Return)
    assert combo.currentData() == value


def _send_wheel(board, position, delta: int):
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
    from PyQt5 import QtCore, QtGui, QtWidgets

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(position),
        QtCore.Qt.NoButton,
        button,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(board, event)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    return zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("b3-repository") / "workspace",
    )


def test_console_cards_answer_zoom_home_and_clim_from_human_controls(experiment) -> None:
    from PyQt5 import QtCore, QtTest, QtWidgets

    import test_u02c_qt_curve_interaction as curve_fixtures
    import test_zlc_qt_image_interaction as image_fixtures

    application = QtWidgets.QApplication.instance()
    console = experiment.task_console()
    try:
        application.processEvents()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # Add the real 2-D and 1-D cards exactly through the header controls.
        assert console.kind_combo.currentData() == "2d"
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        _select_combo_item(console.kind_combo, "1d", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert [card.config.kind for card in console.cards] == ["2d", "1d"]

        # Arm before either raster surface exists.  First-frame construction
        # must replay this visible switch state rather than silently staying off.
        QtTest.QTest.mouseClick(console.selectors_switch, QtCore.Qt.LeftButton)
        assert console.selectors_switch.isChecked()
        image_card, curve_card = console.cards
        image_card._build_plot()
        curve_card._build_plot()
        assert image_card.board._selectors_on
        assert curve_card.board._selectors_on

        # Seed one already-produced curve frame, then zoom through a real wheel
        # event.  SinglePanelHost forwards the typed candidate to PanelCard;
        # its display state must carry that exact view and revision.
        curve_panel = curve_fixtures._curve_panel(0)
        curve_card.board.present_panel(
            curve_panel.raster, curve_panel.display_payload)
        curve_board = curve_card.board.board
        curve_board.setFixedSize(320, 180)
        application.processEvents()
        assert curve_board.selectors_enabled
        target = curve_fixtures._curve_target(curve_board)
        forwarded_views = []
        curve_card.board.viewCommitted.connect(forwarded_views.append)
        center = curve_fixtures._point(target.plot, 0.5, 0.5)
        assert curve_board._numeric_target_at(QtCore.QPointF(center)) is not None
        assert _send_wheel(
            curve_board,
            center,
            -120,
        ).isAccepted()
        assert forwarded_views
        assert forwarded_views[-1].origin == curve_board.visible_curve_origin()
        zoomed = curve_card._display_state()
        assert zoomed.x_view is not None
        assert zoomed.revision == 1
        assert curve_board.curve_selector_fault is None
        accepted = replace(
            curve_panel.display_payload.viewport,
            x_limits=tuple(zoomed.x_view),
            display_revision=zoomed.revision,
        )

        # Echo the accepted revision just as the worker composer does, then the
        # operator's double-middle Home gesture must remove the authored pin.
        curve_card.board.present_panel(
            curve_panel.raster,
            replace(curve_panel.display_payload, viewport=accepted),
        )
        curve_board.setFixedSize(320, 180)
        application.processEvents()
        target = curve_fixtures._curve_target(curve_board)
        QtTest.QTest.mouseDClick(
            curve_board,
            QtCore.Qt.MiddleButton,
            pos=curve_fixtures._point(target.plot, 0.5, 0.5),
        )
        home = curve_card._display_state()
        assert home.x_view is None
        assert home.revision > zoomed.revision

        # A real drag of the image colour rail lands in the one persisted fixed
        # limits writer shared by Setting/Edit; no parallel transient clim fact.
        image_panel = image_fixtures._frame(0).panels[0]
        image_card.board.present_panel(
            image_panel.raster, image_panel.display_payload)
        image_board = image_card.board.board
        image_board.setFixedSize(320, 220)
        application.processEvents()
        assert image_board.selectors_enabled
        binding = image_board._image_bindings[image_card.panel_id]
        rail, *_rest, payload = image_board._clim_rail_target(binding)
        domain = image_board._color_rail_domain(payload)
        low, high = payload.color_limits
        target_low = low + 0.20 * (high - low)
        start = QtCore.QPoint(
            rail.center().x(),
            int(round(image_board._rail_y(low, domain, rail))),
        )
        end = QtCore.QPoint(
            rail.center().x(),
            int(round(image_board._rail_y(target_low, domain, rail))),
        )
        QtTest.QTest.mousePress(image_board, QtCore.Qt.LeftButton, pos=start)
        _drag_move(image_board, end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(image_board, QtCore.Qt.LeftButton, pos=end)

        assert image_card.config.params["relim"] == "fixed"
        assert image_card.config.params["fixed_lo"] == pytest.approx(
            target_low, abs=35.0)
        assert image_card.config.params["fixed_hi"] == high
    finally:
        assert console.shutdown()
        console.window().close()
        application.processEvents()
