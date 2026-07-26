"""Human dot-cycle coverage on the current formal Pulse editor."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtTest

from gui_user_flow import close_pulse_editor
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_workbench.pulse_editor.app import open_pulse_editor


def _until(application, predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


@pytest.fixture
def editor():
    application = ensure_qt_app()
    body = open_pulse_editor()
    body.window().show()
    _until(application, lambda: bool(body.schedule_view.period_cards()))
    yield application, body
    close_pulse_editor(application, body)


def _first_card(body):
    return body.schedule_view.period_cards()[0]


def test_duration_dot_cycles_none_scan_api_off_through_visible_control(editor):
    application, body = editor
    field = _first_card(body).duration_edit
    assert not field.dot.isChecked() and not getattr(field.dot, "_api", False)

    QtTest.QTest.mouseClick(field.dot, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: len(body.current_document.scan_parameters) == 1
        and _first_card(body).duration_edit.dot.isChecked(),
    )
    field = _first_card(body).duration_edit
    assert field.dot.isChecked() and not getattr(field.dot, "_api", False)
    assert field.isReadOnly()
    assert field.text() == "s0"

    QtTest.QTest.mouseClick(field.dot, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: not body.current_document.scan_parameters
        and len(body.current_document.api_parameters) == 1
        and getattr(_first_card(body).duration_edit.dot, "_api", False),
    )
    field = _first_card(body).duration_edit
    assert getattr(field.dot, "_api", False) and not field.dot.isChecked()
    assert not field.isReadOnly()

    QtTest.QTest.mouseClick(field.dot, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: not body.current_document.api_parameters
        and not getattr(_first_card(body).duration_edit.dot, "_api", False),
    )
    field = _first_card(body).duration_edit
    assert not field.dot.isChecked() and not getattr(field.dot, "_api", False)


def test_delay_dot_cycles_none_api_none_through_visible_control(editor):
    application, body = editor

    def delay_field():
        panel = body.schedule_view.channel_panel
        return panel.delay_edits[next(iter(panel.delay_edits))]

    field = delay_field()
    assert not getattr(field.dot, "_api", False)

    QtTest.QTest.mouseClick(field.dot, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: len(body.current_document.api_parameters) == 1
        and getattr(delay_field().dot, "_api", False),
    )
    field = delay_field()
    assert getattr(field.dot, "_api", False)
    assert not field.isReadOnly()

    QtTest.QTest.mouseClick(field.dot, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: not body.current_document.api_parameters
        and not getattr(delay_field().dot, "_api", False),
    )
    assert not getattr(delay_field().dot, "_api", False)
