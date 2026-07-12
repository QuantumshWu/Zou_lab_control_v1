from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_pulse import (
    PORT_DIGITAL,
    PulsePortSpec,
    PulseTarget,
    load_pulse_target,
)
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
        self.fail_layout = False

    def start(self):
        self.events.append("start")
        return self

    def clear_host_config(self):
        self.events.append("clear")

    def check_register_layout(self):
        self.events.append("layout")
        if self.fail_layout:
            raise RuntimeError("layout mismatch")

    def transport_self_test(self):
        self.events.append("transport-self-test")
        if self.fail_self_test:
            raise RuntimeError("self test failed")

    def safe_state(self):
        self.events.append("safe")

    def request_interrupt(self):
        pass

    def close(self):
        self.events.append("close")

    def prepare(self, artifact):
        self.events.append("prepare")

    def fire(self, artifact):
        self.events.append("fire")

    def wait_done(self, artifact, timeout=None):
        self.events.append("wait")
        return True

    def snapshot(self):
        return {"transport": "test"}


def _target():
    return load_pulse_target(ROOT / "pulses" / "deployed_target.json")


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
    bring_up_frozen_session(session)
    assert session.events == ["start", "layout", "clear", "transport-self-test", "safe"]


def test_failed_bringup_attempts_safe_then_closes():
    session = AppSession(StreamerParams())
    session.fail_self_test = True
    with pytest.raises(RuntimeError, match="self test failed"):
        bring_up_frozen_session(session)
    assert session.events[-2:] == ["safe", "close"]


def test_layout_mismatch_closes_without_any_geometry_dependent_write():
    session = AppSession(StreamerParams())
    session.fail_layout = True
    with pytest.raises(RuntimeError, match="layout mismatch"):
        bring_up_frozen_session(session)
    assert session.events == ["start", "layout", "close"]


def test_server_rejects_target_or_session_geometry_drift():
    params = StreamerParams()
    target = _target()
    validate_deployed_target(target, params)

    short_target = PulseTarget(
        ("only",),
        (
            PulsePortSpec(
                "only", PORT_DIGITAL, ("only",), "only", None, 1, "binary", 0, None
            ),
        ),
    )
    with pytest.raises(ValueError, match="raw lane count"):
        validate_deployed_target(short_target, params)

    all_digital = PulseTarget(
        target.raw_lanes,
        tuple(
            PulsePortSpec(
                lane,
                PORT_DIGITAL,
                (lane,),
                lane,
                None,
                1,
                "binary",
                0,
                None,
            )
            for lane in target.raw_lanes
        ),
    )
    with pytest.raises(ValueError, match="digital-port count"):
        validate_deployed_target(all_digital, params)

    renamed = PulseTarget(
        target.raw_lanes,
        (replace(target.ports[0], key="renamed-ch00"), *target.ports[1:]),
    )
    with pytest.raises(ValueError, match="approved deployed topology"):
        validate_deployed_target(renamed, params)

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
    events = list(session.events)
    runtime.close()
    assert session.events == events


def test_rpc_server_construction_failure_closes_the_hardware_owner(monkeypatch, tmp_path):
    params = StreamerParams()
    session = AppSession(params)
    monkeypatch.setattr(
        "zlc_pulse.server_app.open_deployed_session",
        lambda *args, **kwargs: session,
    )

    def fail_serve(*args, **kwargs):
        raise RuntimeError("cannot bind rpc socket")

    monkeypatch.setattr(
        "zlc_pulse.server_app.serve_pulse_execution_service",
        fail_serve,
    )
    with pytest.raises(RuntimeError, match="cannot bind"):
        build_server_runtime(
            _target(),
            backend="jtag-axi",
            state_dir=tmp_path,
            params=params,
            clock_hz=50e6,
        )
    assert session.events[-2:] == ["safe", "close"]


def test_runtime_close_failure_is_not_latched_as_success():
    from zlc_pulse.server_app import PulseServerRuntime

    class FlakySession(AppSession):
        def __init__(self):
            super().__init__(StreamerParams())
            self.fail_once = True
            self.close_fail_once = True

        def safe_state(self):
            self.events.append("safe")
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("safe verification failed")

        def close(self):
            self.events.append("close")
            if self.close_fail_once:
                self.close_fail_once = False
                raise RuntimeError("transport revocation failed")

    class RPCServer:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    session = FlakySession()
    rpc_server = RPCServer()

    class Service:
        def safe_state(self):
            session.safe_state()

    runtime = PulseServerRuntime(Service(), rpc_server, session)

    with pytest.raises(RuntimeError, match="transport revocation failed"):
        runtime.close()
    assert not runtime._closed
    runtime.close()
    assert runtime._closed
    assert rpc_server.close_count == 1
