"""Session device lifecycle is owned by the installation runtime, never QWidget hooks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na


def test_close_stops_authoritative_runs_before_devices_close():
    exp = na.connect("virtual")
    controller = exp.timing.bind_pulse(exp.sequence)
    controller.on_pulse(repeat_forever=True, wait=False)
    runtime = exp._zlc_runtime_services
    assert runtime.resources.active_claims()

    observed = []
    real_close = exp.devices.close

    def tracked_close():
        observed.append((runtime.closed, bool(runtime.resources.active_claims())))
        return real_close()

    exp.devices.close = tracked_close
    exp.close()
    assert observed == [(True, False)]


def test_load_config_stops_affected_authority_before_old_device_close():
    exp = na.connect("virtual")
    old_devices = exp.devices
    old_runtime = exp._zlc_runtime_services
    controller = exp.timing.bind_pulse(exp.sequence)
    controller.on_pulse(repeat_forever=True, wait=False)
    assert old_runtime.resources.active_claims()

    observed = []
    real_close = old_devices.close

    def tracked_close():
        observed.append(bool(old_runtime.resources.active_claims()))
        return real_close()

    old_devices.close = tracked_close
    try:
        exp.load_config("virtual")
        assert observed == [False]
        assert exp.devices is not old_devices
        assert exp._zlc_runtime_services is old_runtime
    finally:
        exp.close()


def test_load_config_open_devices_publishes_the_identity_verified_staged_registry(
    monkeypatch,
):
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera
    from zlc_workbench.legacy_runtime import LegacyDeviceRegistry

    exp = na.connect("virtual")
    events = []
    established = []
    real_establish = LegacyDeviceRegistry.establish

    monkeypatch.setattr(
        VirtualCamera,
        "is_open",
        property(lambda self: bool(getattr(self, "_test_connection_open", False))),
    )

    def open_camera(self):
        events.append("open")
        self._test_connection_open = True
        return self

    def establish(registry, device):
        if type(device) is VirtualCamera:
            assert device.is_open
            events.append("identity")
        binding = real_establish(registry, device)
        if type(device) is VirtualCamera:
            established.append(binding)
        return binding

    monkeypatch.setattr(VirtualCamera, "open", open_camera)
    monkeypatch.setattr(LegacyDeviceRegistry, "establish", establish)
    try:
        exp.load_config("virtual", open_devices=True)
        camera = exp.devices.camera
        runtime = exp._zlc_runtime_services
        assert events == ["open", "identity"]
        assert runtime.registry.has_binding(camera)
        assert runtime.registry.binding_for(camera) is established[0]
    finally:
        exp.close()


def test_shutdown_waits_for_inflight_connection_establishment(monkeypatch):
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera

    exp = na.connect("virtual")
    runtime = exp._zlc_runtime_services
    camera = exp.devices.camera
    entered = threading.Event()
    release = threading.Event()
    establish_done = threading.Event()
    shutdown_done = threading.Event()
    results = {}

    monkeypatch.setattr(
        VirtualCamera,
        "is_open",
        property(lambda self: bool(getattr(self, "_test_connection_open", False))),
    )

    def blocking_open(self):
        entered.set()
        assert release.wait(1.0)
        self._test_connection_open = True
        return self

    monkeypatch.setattr(VirtualCamera, "open", blocking_open)

    def establish():
        try:
            results["establish"] = runtime.ensure_connections((camera,))
        finally:
            establish_done.set()

    def shutdown():
        try:
            results["shutdown"] = runtime.shutdown(timeout=1.0)
        finally:
            shutdown_done.set()

    worker = threading.Thread(target=establish)
    closer = threading.Thread(target=shutdown)
    worker.start()
    assert entered.wait(1.0)
    closer.start()
    assert not shutdown_done.wait(0.05)
    release.set()
    worker.join(1.0)
    closer.join(1.0)

    assert establish_done.is_set()
    assert shutdown_done.is_set()
    assert results == {"establish": True, "shutdown": True}
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.ensure_connections((camera,))
    exp.devices.close()


def test_bad_config_leaves_active_authority_untouched():
    exp = na.connect("virtual")
    controller = exp.timing.bind_pulse(exp.sequence)
    controller.on_pulse(repeat_forever=True, wait=False)
    runtime = exp._zlc_runtime_services
    old_devices = exp.devices
    try:
        with pytest.raises(Exception):
            exp.load_config({"camera": {"type": "no_such_device_type"}})
        assert exp.devices is old_devices
        assert runtime.resources.active_claims()
        assert old_devices.sequencer.snapshot()["state"] == "running"
    finally:
        controller.stop()
        exp.close()


def test_load_config_refuses_close_while_authority_reports_pending_owner(monkeypatch):
    exp = na.connect("virtual")
    old_devices = exp.devices
    runtime = exp._zlc_runtime_services
    monkeypatch.setattr(
        runtime.fence,
        "stop_nodes_using",
        lambda *_args, **_kwargs: (SimpleNamespace(terminated=False),),
    )
    try:
        with pytest.raises(RuntimeError, match="runtime ownership is still cancelling"):
            exp.load_config("virtual")
        assert exp.devices is old_devices
    finally:
        monkeypatch.undo()
        exp.close()


def test_load_config_from_non_qt_thread_does_not_call_qwidget_lifecycle_hooks():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    exp = na.connect("virtual")
    console = exp.task_console()
    result = []

    def swap():
        try:
            exp.load_config("virtual")
            result.append("ok")
        except BaseException as exc:  # pragma: no cover - assertion reports the exact failure
            result.append(exc)

    worker = threading.Thread(target=swap, daemon=False)
    worker.start()
    # Deliberately do not process Qt events while the swap runs.  GUI progress cannot be
    # a correctness dependency or a veto on hardware quiescence.
    worker.join(3.0)
    try:
        assert not worker.is_alive()
        assert result == ["ok"]
        assert console._current_runtime_fence() is exp._zlc_runtime_services
    finally:
        console.shutdown(timeout=1.0)
        exp.close()


def test_load_config_becomes_unavailable_when_old_close_crosses_irreversible_boundary():
    exp = na.connect("virtual")
    old_devices = exp.devices
    real_close = old_devices.close

    def fail_close():
        raise RuntimeError("old device close failed")

    old_devices.close = fail_close
    try:
        with pytest.raises(RuntimeError, match="old device close failed"):
            exp.load_config("virtual")
        assert exp.devices is not old_devices
        with pytest.raises(RuntimeError, match="crossed the old-device close boundary"):
            _ = exp.camera
    finally:
        old_devices.close = real_close
        old_devices.close()
        exp.close()


def test_load_config_stages_all_derived_state_before_publishing(monkeypatch):
    import Zou_lab_control.neutral_atom.devices as devices_module

    exp = na.connect("virtual")
    old_devices = exp.devices
    old_sequence = exp.sequence
    old_readout = exp.readout
    old_timing = exp.timing
    old_calibration = object()
    exp._calibration = old_calibration

    real_load_devices = devices_module.load_devices
    staged = []

    def tracked_load_devices(*args, **kwargs):
        replacement = real_load_devices(*args, **kwargs)
        close_calls = []
        real_close = replacement.close

        def tracked_close():
            close_calls.append(1)
            return real_close()

        replacement.close = tracked_close
        staged.append((replacement, close_calls))
        return replacement

    monkeypatch.setattr(devices_module, "load_devices", tracked_load_devices)

    def fail_derived_state(devices, **_kwargs):
        assert devices is staged[0][0]
        raise RuntimeError("sequence staging failed")

    monkeypatch.setattr(exp, "_build_imaging_sequence_for_devices", fail_derived_state)
    try:
        with pytest.raises(RuntimeError, match="sequence staging failed"):
            exp.load_config("virtual")
        assert exp.devices is old_devices
        assert exp.sequence is old_sequence
        assert exp.readout is old_readout
        assert exp.timing is old_timing
        assert exp._calibration is old_calibration
        assert len(staged) == 1 and staged[0][1] == [1]
    finally:
        exp.close()
