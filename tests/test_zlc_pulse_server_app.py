from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from conftest import private_pulse_backend_snapshot, pulse_backend_completion_for

from fpga.pulse_streamer.host.image import StreamerParams
import zlc_pulse.deployment as pulse_deployment
from zlc_pulse import (
    PORT_DIGITAL,
    PulseDocument,
    PulseExecutionForm,
    PulsePeriod,
    PulsePortSpec,
    PulseTarget,
    compile_pulse_artifact,
    load_deployed_geometry_facts,
    load_deployed_pulse_target,
    load_pulse_target,
    pack_target_ir,
    pulse_target_manifest_from_xdc,
    require_deployed_geometry_facts,
    validate_target_ir_for_geometry,
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
        self.state = "IDLE"
        self.prepared = None
        self.scan_points = 0

    def start(self):
        self.events.append("start")
        return self

    def clear_host_config(self):
        self.events.append("clear")
        self.prepared = None
        self.state = "SAFE"

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
        self.prepared = None
        self.state = "SAFE"

    def request_interrupt(self):
        pass

    def close(self):
        self.events.append("close")
        self.state = "CLOSED"

    def prepare(self, artifact):
        self.events.append("prepare")
        self.prepared = artifact
        self.scan_points = len(artifact.target_ir.scan_points)
        self.state = "PREPARED"

    def fire(self, artifact):
        self.events.append("fire")
        self.state = "RUNNING"

    def await_completion(self, artifact, timeout=None):
        self.events.append("wait")
        self.state = "DONE"
        return pulse_backend_completion_for(artifact)

    def snapshot(self):
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=len(_target().raw_lanes),
            artifact=self.prepared,
            scan_point_count=self.scan_points,
        )


def _target():
    return load_deployed_pulse_target()


def _manifest():
    return pulse_target_manifest_from_xdc(
        _target(),
        ROOT / "fpga" / "board_config" / "board.xdc",
    )


def test_target_file_loader_accepts_only_the_current_canonical_schema(tmp_path):
    target = _target()
    path = tmp_path / "pulse-target.json"
    path.write_text(json.dumps(pulse_target_to_tree(target)), encoding="utf-8")
    assert load_pulse_target(path) == target

    path.write_text(json.dumps({"schema": "old-target", "channels": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pulse_target(path)


def test_deployment_is_the_single_strict_owner_of_geometry_facts(
    tmp_path,
    monkeypatch,
):
    canonical_source = ROOT / "fpga" / "board_config" / "streamer_config.json"
    alternate_source = tmp_path / "alternate-streamer-config.json"
    alternate_source.write_text(
        canonical_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cwd_source = tmp_path / "fpga" / "board_config" / "streamer_config.json"
    cwd_source.parent.mkdir(parents=True)
    cwd_source.write_text(
        canonical_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZLC_PS_CONFIG", str(alternate_source))
    monkeypatch.chdir(tmp_path)

    facts = load_deployed_geometry_facts()

    assert facts.clock_hz == 50e6
    assert facts.source == canonical_source.resolve()
    assert load_deployed_geometry_facts(alternate_source).source == (
        alternate_source.resolve()
    )
    assert require_deployed_geometry_facts(
        facts.geometry_fingerprint,
        facts.clock_hz,
    ) == facts.source
    with pytest.raises(ValueError, match="geometry differs"):
        require_deployed_geometry_facts(
            facts.geometry_fingerprint ^ 1,
            facts.clock_hz,
        )
    with pytest.raises(ValueError, match="clock differs"):
        require_deployed_geometry_facts(
            facts.geometry_fingerprint,
            facts.clock_hz / 2,
        )


def test_default_compile_pack_and_validation_use_the_checked_deployment(
    tmp_path,
    monkeypatch,
):
    target = _target()
    low = tuple(0 for _ in target.raw_lanes)
    document = PulseDocument(
        name="strict deployed geometry witness",
        target=target,
        time_step_ns=20.0,
        periods=(PulsePeriod("safe", 20.0, "ns", "safe", low),),
    )
    canonical = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        params=StreamerParams(),
    )
    config = json.loads(
        (ROOT / "fpga" / "board_config" / "streamer_config.json").read_text(
            encoding="utf-8"
        )
    )
    config["params"]["channel_count"] -= 1
    alternate = tmp_path / "narrower-streamer-config.json"
    alternate.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(pulse_deployment, "DEFAULT_CONFIG_PATH", alternate)

    with pytest.raises(ValueError, match="more raw lanes"):
        compile_pulse_artifact(
            document,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
        )
    with pytest.raises(ValueError, match="channels"):
        validate_target_ir_for_geometry(canonical.target_ir)
    with pytest.raises(ValueError, match="channels"):
        pack_target_ir(canonical.target_ir)


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
        build_service_for_session(_manifest(), session, params=params, clock_hz=50e6)


def test_server_runtime_composes_only_the_current_service(monkeypatch, tmp_path):
    params = StreamerParams()
    session = AppSession(params)
    calls: list[tuple[str, object]] = []
    target_validation_count = 0

    def count_target_validation(target, geometry):
        nonlocal target_validation_count
        target_validation_count += 1
        validate_deployed_target(target, geometry)

    monkeypatch.setattr(
        "zlc_pulse.server_app.validate_deployed_target",
        count_target_validation,
    )

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
        _manifest(),
        backend="jtag-axi",
        state_dir=tmp_path,
        params=params,
        clock_hz=50e6,
        host="127.0.0.1",
        port=18862,
    )

    assert calls[0][0] == "serve"
    assert calls[0][1][1:] == ("127.0.0.1", 18862, False)
    assert target_validation_count == 1
    assert runtime.service.snapshot().physical_state == "SAFE"
    assert runtime.service.snapshot().safe_readback_confirmed
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
            _manifest(),
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

    with pytest.raises(RuntimeError, match="safe verification failed"):
        runtime.close()
    assert not runtime._closed
    with pytest.raises(RuntimeError, match="transport revocation failed"):
        runtime.close()
    assert not runtime._closed
    runtime.close()
    assert runtime._closed
    assert rpc_server.close_count == 1
    assert session.events == ["safe", "safe", "close", "safe", "close"]
