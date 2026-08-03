"""Formal Pulse Preview is a thin Qt composition over the one zlc_plot host."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

from gui_user_flow import close_pulse_editor
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_plot import RasterPlotHost


ROOT = Path(__file__).parents[1]
PULSE_PATH = ROOT / "pulses" / "imaging_template.json"


def _until(application, predicate, *, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _click_tab(body, page) -> None:
    index = body.tabs.indexOf(page)
    assert index >= 0
    bar = body.tabs.tabBar()
    QtTest.QTest.mouseClick(
        bar,
        QtCore.Qt.LeftButton,
        pos=bar.tabRect(index).center(),
    )


def _choose_combo(application, combo, text: str) -> None:
    index = combo.findText(text)
    assert index >= 0
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    _until(application, lambda: combo.view().isVisible())
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _position in range(index):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    _until(application, lambda: combo.currentText() == text)


@pytest.fixture
def preview_body(tmp_path):
    application = ensure_qt_app()
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    body = open_pulse_editor(
        path=PULSE_PATH.name,
        pulses_root=PULSE_PATH.parent,
        output_root=tmp_path / "output",
    )
    _until(application, lambda: body.window() is not body and body.window().isVisible())
    _click_tab(body, body.preview_view)
    _until(
        application,
        lambda: (
            body.tabs.currentWidget() is body.preview_view
            and body.preview_widget is not None
            and body.preview_widget.presented_front is not None
            and body.worker_idle
        ),
    )
    yield body
    close_pulse_editor(application, body)
    assert body.worker_idle


def test_preview_mounts_the_public_zlc_plot_surface_with_board_labels(preview_body):
    body = preview_body
    assert isinstance(body.preview_host, RasterPlotHost)
    assert body.preview_widget is not None
    assert body.preview_widget.host is body.preview_host
    assert body.preview_widget.parent() is body.preview_view.preview_body
    assert re.fullmatch(
        r"\d+/\d+ plotted \((active|all) channels\)"
        r" \| repeat (?:∞(?: \+ P\d+-P\d+ x\d+)?|P\d+-P\d+ x\d+)",
        body.preview_view.preview_status.text(),
    )
    projected = body._controller.preview_update().plot
    assert projected is not None
    rows = tuple(
        (channel.channel_id, channel.label) for channel in projected.data.channels
    ) + tuple(
        (trace.name, trace.label) for trace in projected.data.analog_traces
    )
    assert rows
    assert any(key != label for key, label in rows)


def test_show_all_and_size_reconcile_on_the_same_widget(preview_body):
    application = ensure_qt_app()
    body = preview_body
    widget = body.preview_widget
    host = body.preview_host
    initial = body._controller.preview_update().plot
    assert widget is not None and host is not None and initial is not None

    QtTest.QTest.mouseClick(
        body.preview_view.preview_include_off,
        QtCore.Qt.LeftButton,
    )
    _until(
        application,
        lambda: (
            body._controller.preview_update().plot is not None
            and body._controller.preview_update().plot.include_off_rows
            and body.worker_idle
        ),
    )
    expanded = body._controller.preview_update().plot
    assert expanded is not None
    assert body.preview_widget is widget
    assert body.preview_host is host
    assert len(expanded.data.channels) >= len(initial.data.channels)

    _choose_combo(application, body.preview_view.preview_size_combo, "4x4")
    _until(
        application,
        lambda: (
            body.preview_widget is widget
            and widget.presented_front is not None
            and widget.presented_front.identity.preset == "4x4"
            and body.worker_idle
        ),
    )
    assert body.preview_view.preview_size_pinned


def test_preview_selector_input_is_owned_by_the_zlc_plot_widget(preview_body):
    application = ensure_qt_app()
    body = preview_body
    widget = body.preview_widget
    assert widget is not None and widget.presented_front is not None
    QtTest.QTest.mouseClick(
        body.preview_view.preview_selectors_switch,
        QtCore.Qt.LeftButton,
    )
    assert widget.interaction_enabled
    before = widget.presented_front
    centre = widget.rect().center()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(centre),
        QtCore.QPointF(widget.mapToGlobal(centre)),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, wheel)
    _until(
        application,
        lambda: (
            widget.presented_front is not None
            and widget.presented_front.identity.sequence > before.identity.sequence
            and body.worker_idle
        ),
    )
    assert (
        widget.presented_front.interaction.axes[0].x_limits
        != before.interaction.axes[0].x_limits
    )


def test_save_figure_exports_the_same_host_surface(preview_body, tmp_path, monkeypatch):
    application = ensure_qt_app()
    body = preview_body
    target = tmp_path / "operator-preview.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "Pulse figure (*.png)"),
    )
    QtTest.QTest.mouseClick(
        body.preview_view.preview_save_figure_button,
        QtCore.Qt.LeftButton,
    )
    _until(
        application,
        lambda: target.exists() and target.stat().st_size > 0 and body.worker_idle,
    )
    image = QtGui.QImage(str(target))
    assert not image.isNull()
