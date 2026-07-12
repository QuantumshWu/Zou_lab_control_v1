from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_pulse import load_pulse_document, load_pulse_target
from zlc_pulse.server_app import (
    bring_up_frozen_session,
    build_server_runtime,
    build_service_for_session,
    validate_deployed_target,
)
from zlc_pulse.target import pulse_target_to_tree


ROOT = Path(__file__).parents[1]


class AppSession:
    def __init__(self, params: StreamerParams, clock_hz: float = 50e6) -> None:
        self.params = params
        self.clock_hz = clock_hz
        self.events: list[str] = []
        self.fail_self_test = False

    def start(self):
        self.events.append("start")
        return self

    def clear_host_config(self):
        self.events.append("clear")

    def check_register_layout(self):
        self.events.append("layout")

    def axi_self_test(self):
        self.events.append("axi-self-test")
        if self.fail_self_test:
            raise RuntimeError("self test failed")

    def link_self_test(self):
        self.events.append("uart-self-test")

    def safe_state(self):
        self.events.append("safe")

    def close(self):
        self.events.append("close")

    def prepare_compiled_artifact(self, artifact):
        self.events.append("prepare")

    def fire_compiled_artifact(self, artifact):
        self.events.append("fire")

    def wait_done_compiled_artifact(self, artifact, timeout=None):
        self.events.append("wait")
        return True

    def current_snapshot(self):
        return {"transport": "test"}


def _target():
    return load_pulse_document(ROOT / "pulses" / "camera_imaging_address_switch.json").target


def test_target_file_loader_accepts_only_the_current_canonical_schema(tmp_path):
    target = _target()
    path = tmp_path / "pulse-target.json"
    path.write_text(json.dumps(pulse_target_to_tree(target)), encoding="utf-8")
    assert load_pulse_target(path) == target

    path.write_text(json.dumps({"schema": "old-target", "channels": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pulse_target(path)


def test_frozen_bringup_is_safe_and_never_programs_hardware():
    session = AppSession(StreamerParams())
    bring_up_frozen_session(session, "jtag-axi")
    assert session.events == ["start", "clear", "layout", "axi-self-test", "safe"]


def test_failed_bringup_attempts_safe_then_closes():
    session = AppSession(StreamerParams())
    session.fail_self_test = True
    with pytest.raises(RuntimeError, match="self test failed"):
        bring_up_frozen_session(session, "jtag-axi")
    assert session.events[-2:] == ["safe", "close"]


def test_server_rejects_target_or_session_geometry_drift():
    params = StreamerParams()
    target = _target()
    validate_deployed_target(target, params)

    short_target = load_pulse_document(ROOT / "pulses" / "imaging_template.json").target
    with pytest.raises(ValueError, match="raw lane count"):
        validate_deployed_target(short_target, params)

    session = AppSession(replace(params, bank_size=1024))
    with pytest.raises(ValueError, match="session geometry"):
        build_service_for_session(target, session, params=params, clock_hz=50e6)


def test_server_runtime_composes_only_the_current_service(monkeypatch, tmp_path):
    params = StreamerParams()
    session = AppSession(params)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "zlc_pulse.server_app.open_deployed_session",
        lambda *args, **kwargs: session,
    )

    class RPCServer:
        def close(self):
            calls.append(("close", None))

    def fake_serve(service, *, host, port, start):
        calls.append(("serve", (service, host, port, start)))
        return RPCServer()

    monkeypatch.setattr("zlc_pulse.server_app.serve_pulse_execution_service", fake_serve)
    runtime = build_server_runtime(
        _target(),
        backend="jtag-axi",
        state_dir=tmp_path,
        params=params,
        clock_hz=50e6,
        host="127.0.0.1",
        port=18862,
    )

    assert calls[0][0] == "serve"
    assert calls[0][1][1:] == ("127.0.0.1", 18862, False)
    assert runtime.service.snapshot()["backend"]["transport"] == "test"
    runtime.close()
    assert calls[-1] == ("close", None)
    assert session.events[-2:] == ["safe", "close"]
