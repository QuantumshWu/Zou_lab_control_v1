"""Formal human-input flows for the ordered DeviceInstance manager."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

from Zou_lab_control.api import WorkspacePaths, connect, device_manager
from Zou_lab_control.workbench._composition import ExperimentDeviceAdmin
from tests.gui_user_flow import (
    capture_offscreen_window,
    choose_combo_data,
    configure_offscreen_fast_path,
    until,
    widget_gone,
)
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    installation_template,
    save_installation_config,
)
from zlc_neutral_atom.device_types import device_type, discover_device_types
from zlc_neutral_atom.runtime.run import CancelOutcome
from zlc_pulse import PulseExecutionForm, load_pulse_document


ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "pulses" / "imaging_template.json"


def _replace_text(widget, text: str) -> None:
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(widget, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(widget, text)


def _workspace(tmp_path) -> WorkspacePaths:
    return WorkspacePaths.for_workspace((tmp_path / "workspace").resolve())


def test_formal_device_graph_edit_save_load_apply_and_close(tmp_path, monkeypatch):
    configure_offscreen_fast_path()
    application = ensure_qt_app()
    body = device_manager(workspace=_workspace(tmp_path))
    wrapper = body.window()
    try:
        assert body.tabs.tabText(0) == "Config"
        assert tuple(body._device_cards) == (
            "sequencer",
            "rf",
            "camera",
            "mot-camera",
        )
        stable_cards = {key: id(value) for key, value in body._device_cards.items()}
        seed_widget = body._device_cards["camera"].form.widget_for("seed")
        stable_seed_widget = id(seed_widget)
        sequencer_ref = body._device_cards["camera"].form.widget_for(
            "sequencer_ref"
        )
        assert isinstance(sequencer_ref, QtWidgets.QComboBox)
        assert tuple(
            sequencer_ref.itemData(index)
            for index in range(sequencer_ref.count())
        ) == ("sequencer",)

        _replace_text(seed_widget, "11")
        until(
            application,
            lambda: body._controller.editor.row("camera").parameters["seed"] == 11,
        )
        assert body._controller.editor.row("mot-camera").parameters["seed"] == 7
        assert id(body._device_cards["camera"].form.widget_for("seed")) == stable_seed_widget
        assert {key: id(value) for key, value in body._device_cards.items()} == stable_cards

        before = set(body._device_cards)
        QtTest.QTest.mouseClick(
            body._domain_add_buttons["camera"],
            QtCore.Qt.LeftButton,
        )
        until(application, lambda: len(body._device_cards) == len(before) + 1)
        added = next(iter(set(body._device_cards).difference(before)))
        added_card = body._device_cards[added]
        added_identity = id(added_card)

        choose_combo_data(added_card.type_combo, "camera.pylon", application)
        until(
            application,
            lambda: body._controller.editor.row(added).type_id == "camera.pylon",
        )
        assert id(body._device_cards[added]) == added_identity
        assert {key: id(body._device_cards[key]) for key in stable_cards} == stable_cards

        QtTest.QTest.mouseClick(added_card.remove_button, QtCore.Qt.LeftButton)
        until(application, lambda: added not in body._device_cards)
        assert {key: id(body._device_cards[key]) for key in stable_cards} == stable_cards

        saved = tmp_path / "installation.json"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(saved), ""),
        )
        QtTest.QTest.mouseClick(body.save_as_button, QtCore.Qt.LeftButton)
        until(application, saved.is_file)
        assert not body._controller.editor.dirty

        _replace_text(seed_widget, "12")
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(saved), ""),
        )
        QtTest.QTest.mouseClick(body.load_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: body._controller.editor.row("camera").parameters["seed"] == 11,
        )
        assert {key: id(body._device_cards[key]) for key in stable_cards} == stable_cards

        QtTest.QTest.mouseClick(body.apply_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: body._controller.state.active_config is not None,
            timeout=15.0,
        )
        until(
            application,
            lambda: set(body._loaded_cards)
            == {"camera", "mot-camera", "rf", "sequencer"},
        )
        capture = capture_offscreen_window(
            application,
            body,
            tmp_path / "device-manager.png",
            settle_ms=100,
        )
        assert capture["image_pixels"]["width"] > 0
        assert {key: id(body._device_cards[key]) for key in stable_cards} == stable_cards
    finally:
        if not body.permanently_closed:
            wrapper.close()
            until(application, lambda: body.permanently_closed, timeout=15.0)


def test_task_console_launcher_composes_once_through_device_manager(tmp_path):
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
    try:
        capture = capture_offscreen_window(
            application,
            devices,
            tmp_path / "task-console-device-init.png",
            settle_ms=100,
        )
        assert capture["image_pixels"]["width"] > 0

        QtTest.QTest.mouseClick(devices.apply_button, QtCore.Qt.LeftButton)
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
        assert flow.console.window().isVisible()
        assert flow.pulse.window().isVisible()

        console_wrapper = flow.console.window()
        console_wrapper.close()
        until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.finish_close(application, timeout_seconds=15.0)
        assert devices.permanently_closed
    finally:
        flow.finish_close(application, timeout_seconds=15.0)


def test_saved_graph_opens_as_the_exact_editing_baseline(tmp_path):
    configure_offscreen_fast_path()
    application = ensure_qt_app()
    path = tmp_path / "installation.json"
    document = installation_template("virtual", seed=23)
    save_installation_config(path, document)

    body = device_manager(path, workspace=_workspace(tmp_path))
    wrapper = body.window()
    try:
        assert body._controller.editor.path == path.resolve()
        assert not body._controller.editor.dirty
        assert body._controller.editor.candidate() == document
        assert body.document_name.text() == path.name
        assert body.save_button.isEnabled()
    finally:
        wrapper.close()
        until(application, lambda: body.permanently_closed, timeout=15.0)


def test_discovered_device_is_separate_and_adds_one_stable_card(
    tmp_path,
    monkeypatch,
):
    import zlc_workbench.device_manager.controller as controller_impl
    import zlc_workbench.device_manager.window as window_impl

    discovered = DeviceInstanceConfig(
        "discovered-mot-camera",
        "discovered_mot_camera",
        "camera.virtual_mot",
        {"sequencer_ref": "sequencer", "seed": 71},
    )
    source = device_type("camera.virtual_mot")
    descriptor = replace(source, discover=lambda: (discovered,))
    descriptors = tuple(
        descriptor if item.type_id == descriptor.type_id else item
        for item in discover_device_types()
    )
    monkeypatch.setattr(controller_impl, "discover_device_types", lambda: descriptors)
    monkeypatch.setattr(window_impl, "discover_device_types", lambda: descriptors)

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    body = device_manager(workspace=_workspace(tmp_path))
    wrapper = body.window()
    try:
        assert "discovered-mot-camera" not in body._device_cards
        QtTest.QTest.mouseClick(body.discover_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: "discovered-mot-camera" in body._discovered_cards,
        )
        discovered_card = body._discovered_cards["discovered-mot-camera"]
        assert "discovered-mot-camera" not in body._device_cards
        QtTest.QTest.mouseClick(discovered_card.add_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: "discovered-mot-camera" in body._device_cards,
        )
        assert body._discovered_cards["discovered-mot-camera"] is discovered_card
        assert not discovered_card.add_button.isEnabled()
    finally:
        wrapper.close()
        until(application, lambda: body.permanently_closed, timeout=15.0)


def test_apply_rejects_an_active_run_without_touching_the_runtime(tmp_path):
    experiment = connect(
        installation_template("virtual", seed=31),
        workspace=_workspace(tmp_path),
        name="active-run-apply",
    )
    admin = ExperimentDeviceAdmin.bound(experiment)
    before_config = experiment.installation_config
    before_runtime = experiment.device_catalog.runtime_instance_id
    document = load_pulse_document(IMAGING_PULSE)
    api_values = {
        parameter.parameter_id: document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    }
    handle = experiment.pulse.start(
        experiment.pulse.request(
            document,
            PulseExecutionForm.CONTINUOUS_MONITOR,
            api_values=api_values,
        )
    )
    try:
        deadline = time.monotonic() + 5.0
        while handle.snapshot().phase != "holding-pulse":
            if time.monotonic() >= deadline:
                raise AssertionError("continuous pulse did not reach its hold phase")
            time.sleep(0.005)

        with pytest.raises(RuntimeError, match="active Run"):
            admin.apply(installation_template("virtual", seed=32))

        assert experiment.installation_config == before_config
        assert experiment.device_catalog.runtime_instance_id == before_runtime
    finally:
        if not handle.snapshot().state.terminal:
            assert handle.cancel("DeviceManager active-Run contract") in {
                CancelOutcome.REQUESTED,
                CancelOutcome.ALREADY_TERMINAL,
            }
        handle.wait(5.0)
        experiment.close()


def test_failed_candidate_restores_the_previous_graph_on_the_same_experiment(
    tmp_path,
):
    experiment = connect(
        installation_template("virtual", seed=41),
        workspace=_workspace(tmp_path),
        name="candidate-rollback",
    )
    admin = ExperimentDeviceAdmin.bound(experiment)
    previous = experiment.installation_config
    before_runtime = experiment.device_catalog.runtime_instance_id
    candidate = installation_template(
        "remote_pulse",
        host="127.0.0.1",
        port=1,
    )
    try:
        with pytest.raises(Exception):
            admin.apply(candidate)

        state = admin.state()
        assert state.active_config == previous
        assert experiment.installation_config == previous
        assert experiment.device_catalog.runtime_instance_id != before_runtime
        assert state.runtime_instance_id == experiment.device_catalog.runtime_instance_id
        assert admin.experiment_for_runtime(state.runtime_instance_id) is experiment
    finally:
        experiment.close()


def test_failed_candidate_and_failed_restore_leave_no_active_session(
    tmp_path,
    monkeypatch,
):
    experiment = connect(
        installation_template("virtual", seed=51),
        workspace=_workspace(tmp_path),
        name="failed-restore",
    )
    admin = ExperimentDeviceAdmin.bound(experiment)

    def fail_connect(*_args, **_kwargs):
        raise ConnectionError("controlled connection failure")

    monkeypatch.setattr("Zou_lab_control.api.facade.connect", fail_connect)
    with pytest.raises(BaseExceptionGroup, match="could not be restored"):
        admin.apply(installation_template("virtual", seed=52))

    state = admin.state()
    assert state.active_config is None
    assert state.runtime_instance_id is None
    assert not state.can_apply
