"""Formal human-input flow for the current DeviceManager product."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from Zou_lab_control.api import WorkspacePaths, device_manager
from tests.gui_user_flow import (
    capture_offscreen_window,
    configure_offscreen_fast_path,
    until,
    widget_gone,
)
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    save_installation_config,
)


def _replace_text(widget, text: str) -> None:
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(widget, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(widget, text)


def _workspace(tmp_path) -> WorkspacePaths:
    return WorkspacePaths.for_workspace((tmp_path / "workspace").resolve())


def test_formal_device_manager_edits_locally_then_initializes_and_closes(
    tmp_path,
    monkeypatch,
):
    configure_offscreen_fast_path()
    application = ensure_qt_app()
    body = device_manager(workspace=_workspace(tmp_path))
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
        until(
            application,
            lambda: body._controller.state.active_config is not None,
            timeout=15.0,
        )
        until(
            application,
            lambda: set(body._loaded_cards)
            == {
                "camera",
                "mot_camera",
                "rf",
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
        until(application, lambda: body.permanently_closed, timeout=15.0)
        assert body._controller.state.active_config is None
    finally:
        if not body.permanently_closed:
            wrapper.close()
            until(application, lambda: body.permanently_closed, timeout=15.0)


def test_task_console_launcher_initializes_through_device_manager_then_reuses_owner(
    tmp_path,
):
    """The standalone GUI enters through formal DeviceManager, not ``connect``."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()

    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--name",
            "launcher-current",
            "--seed",
            "19",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    device_wrapper = devices.window()
    console_wrapper = None
    try:
        device_capture = capture_offscreen_window(
            application,
            devices,
            tmp_path / "task-console-device-init.png",
            settle_ms=100,
        )
        assert device_capture["image_pixels"]["width"] > 0

        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        assert flow.experiment is not None
        assert (
            devices._controller.state.runtime_instance_id
            == flow.experiment.device_catalog.runtime_instance_id
        )
        assert not device_wrapper.isVisible()

        console_wrapper = flow.console.window()
        assert console_wrapper.isVisible()
        console_capture = capture_offscreen_window(
            application,
            flow.console,
            tmp_path / "task-console-after-init.png",
            settle_ms=100,
        )
        assert console_capture["image_pixels"]["width"] > 0

        console_wrapper.close()
        until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        until(application, lambda: devices.permanently_closed, timeout=15.0)
        assert devices._controller.state.active_config is None
    finally:
        if flow.console is not None:
            flow.console.window().close()
        flow.close()
        application.processEvents()


def test_saved_config_opens_as_the_exact_editing_baseline(tmp_path):
    configure_offscreen_fast_path()
    application = ensure_qt_app()
    path = tmp_path / "installation.json"
    document = InstallationConfigDocument.from_parameters("virtual", {"seed": 23})
    save_installation_config(path, document)

    body = device_manager(path, workspace=_workspace(tmp_path))
    wrapper = body.window()
    try:
        assert body._controller.editor.path == path.resolve()
        assert body._controller.editor.baseline_digest == document.content_digest
        assert not body._controller.editor.dirty
        assert body.document_name.text() == path.name
        assert body.save_button.isEnabled()
    finally:
        wrapper.close()
        until(application, lambda: body.permanently_closed, timeout=15.0)
