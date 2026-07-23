"""Formal human-input flow for the current DeviceManager product."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from Zou_lab_control.notebook import device_manager
from tests.gui_user_flow import (
    capture_offscreen_window,
    configure_offscreen_fast_path,
    until,
)
from zlc_frontend.qt_widgets import ensure_qt_app


def _replace_text(widget, text: str) -> None:
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(widget, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(widget, text)


def test_formal_device_manager_edits_locally_then_initializes_and_closes(
    tmp_path,
    monkeypatch,
):
    configure_offscreen_fast_path()
    application = ensure_qt_app()
    body = device_manager(repository=tmp_path / "workspace")
    wrapper = body.window()
    try:
        assert body.tabs.tabText(0) == "Config"
        seed_widget = body.form.widget_for("seed")
        stable_identity = id(seed_widget)

        _replace_text(seed_widget, "11")
        until(application, lambda: body._controller.editor.values["seed"] == 11)
        assert id(body.form.widget_for("seed")) == stable_identity
        assert body._controller.editor.dirty

        saved = tmp_path / "installation.json"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(saved), ""),
        )
        QtTest.QTest.mouseClick(body.save_as_button, QtCore.Qt.LeftButton)
        until(application, lambda: saved.is_file())
        assert not body._controller.editor.dirty

        _replace_text(seed_widget, "12")
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(saved), ""),
        )
        QtTest.QTest.mouseClick(body.load_button, QtCore.Qt.LeftButton)
        until(application, lambda: body._controller.editor.values["seed"] == 11)
        assert id(body.form.widget_for("seed")) == stable_identity

        _replace_text(seed_widget, "13")

        QtTest.QTest.mouseClick(body.lifecycle_button, QtCore.Qt.LeftButton)
        until(application, lambda: body.experiment is not None, timeout=15.0)
        until(
            application,
            lambda: set(body._loaded_cards)
            == {
                "camera",
                "monitor_camera",
                "mot_camera",
                "sequencer",
            },
        )
        capture = capture_offscreen_window(
            application,
            body,
            tmp_path / "device-manager.png",
            settle_ms=100,
        )
        assert capture["image_pixels"]["width"] > 0

        _replace_text(seed_widget, "14")
        until(
            application,
            lambda: body.lifecycle_button.text() == "Shutdown for restart",
        )
        assert id(body.form.widget_for("seed")) == stable_identity

        wrapper.close()
        until(application, lambda: not wrapper.isVisible(), timeout=15.0)
        assert body.experiment is None
    finally:
        if wrapper.isVisible():
            wrapper.close()
            until(application, lambda: not wrapper.isVisible(), timeout=15.0)
