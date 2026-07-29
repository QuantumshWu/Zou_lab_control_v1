from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest
from conftest import private_pulse_backend_snapshot, pulse_backend_completion_for

from fpga.pulse_streamer.host.image import StreamerParams

from zlc_pulse import (
    PulseExecutionForm,
    PulseExecutionService,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    load_pulse_document,
    pulse_server_snapshot_from_tree,
    pulse_server_snapshot_to_tree,
    pulse_target_manifest_from_lanes,
)
from zlc_pulse.server import (
    decode_artifact_message,
    decode_prepared_ref_message,
    encode_completion_message,
    encode_continuous_failure_message,
    encode_prepared_ref_message,
)
from zlc_storage import encode


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


class _InProcessTimedResult:
    """Synchronous stand-in for the RPyC timed-result boundary only."""

    def __init__(self, call):
        self._call = call

    @property
    def value(self):
        return self._call()


def _install_in_process_rpyc_timed(monkeypatch):
    """Keep netref scheduling/timeout behavior in transport integration tests."""

    import rpyc

    def timed(call, timeout):
        assert timeout > 0

        def invoke(*args, **kwargs):
            return _InProcessTimedResult(lambda: call(*args, **kwargs))

        return invoke

    monkeypatch.setattr(rpyc, "timed", timed)


class Backend:
    def __init__(self) -> None:
        self.prepared = None
        self.safe = True
        self.state = "IDLE"
        self.scan_points = 0
        self.safe_calls = 0

    def prepare(self, artifact):
        self.prepared = artifact
        self.safe = False
        self.state = "PREPARED"
        self.scan_points = len(artifact.target_ir.scan_points)

    def fire(self, artifact):
        assert artifact is self.prepared
        self.state = "RUNNING"

    def await_completion(self, artifact, timeout):
        assert artifact is self.prepared
        self.state = "DONE"
        return pulse_backend_completion_for(artifact, transport_id="client-test")

    def wait_continuous_failure(self, artifact, timeout):
        assert artifact is self.prepared
        return None

    def safe_state(self):
        self.safe_calls += 1
        self.prepared = None
        self.safe = True
        self.state = "SAFE"

    def request_interrupt(self):
        pass

    def snapshot(self):
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=len(_fixture_manifest().target.raw_lanes),
            artifact=self.prepared,
            scan_point_count=self.scan_points,
        )


class Root:
    def __init__(self, service: PulseExecutionService) -> None:
        self.service = service

    def current_snapshot(self):
        return encode(pulse_server_snapshot_to_tree(self.service.snapshot()))

    def current_prepare(self, payload):
        return encode_prepared_ref_message(
            self.service.prepare(decode_artifact_message(bytes(payload)))
        )

    def current_fire(self, payload):
        self.service.fire(decode_prepared_ref_message(bytes(payload)))
        return True

    def current_complete(self, payload, timeout):
        return encode_completion_message(
            self.service.complete(
                decode_prepared_ref_message(bytes(payload)),
                timeout=timeout,
            )
        )

    def current_wait_continuous_failure(self, payload, timeout):
        return encode_continuous_failure_message(
            self.service.wait_continuous_failure(
                decode_prepared_ref_message(bytes(payload)),
                timeout=timeout,
            )
        )

    def current_interrupt_safe_state(self, generation):
        return encode(
            pulse_server_snapshot_to_tree(
                self.service.safe_state_for_generation(generation)
            )
        )


class Connection:
    def __init__(self, service: PulseExecutionService) -> None:
        self.root = Root(service)
        self.closed = False

    def close(self):
        self.closed = True


def _fixture():
    document = load_pulse_document(IMAGING_TEMPLATE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    backend = Backend()
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    connection = Connection(service)
    return artifact, service, connection


def _fixture_manifest():
    document = load_pulse_document(IMAGING_TEMPLATE)
    return pulse_target_manifest_from_lanes(document.target)


def test_remote_client_runs_one_current_generation_without_legacy_payloads(monkeypatch):
    _install_in_process_rpyc_timed(monkeypatch)
    artifact, _service, connection = _fixture()
    interrupt_connection = Connection(_service)
    client = RemotePulseExecutionClient(
        connection,
        interrupt_connection,
        transport_timeout_seconds=10.0,
    )

    reference = client.prepare(artifact)
    client.fire(reference)
    completion = client.complete(reference, timeout=1.0)

    assert completion.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert completion.hardware_terminal.transport_id == "client-test"
    assert client.safe_state().state == "SAFE"
    client.close()
    assert connection.closed
    assert interrupt_connection.closed


def test_remote_client_rejects_unbounded_or_stale_completion():
    artifact, service, connection = _fixture()
    client = RemotePulseExecutionClient(
        connection,
        Connection(service),
        transport_timeout_seconds=2.0,
    )
    reference = client.prepare(artifact)
    client.fire(reference)

    with pytest.raises(ValueError, match="transport backstop"):
        client.complete(reference, timeout=2.0)

    service.safe_state()
    service.renew_connection_generation()
    with pytest.raises(RuntimeError, match="generation changed"):
        client.snapshot()


def test_remote_client_rejects_non_current_snapshot_schema():
    class BadRoot:
        def current_snapshot(self):
            return encode({"schema": "old-server"})

        current_prepare = current_fire = current_complete = lambda *args: True
        current_wait_continuous_failure = lambda *args: True

        def current_interrupt_safe_state(self, generation):
            return encode({"schema": "old-server"})

    class BadConnection:
        root = BadRoot()

        def close(self):
            pass

    with pytest.raises(ValueError, match="unknown field set"):
        RemotePulseExecutionClient(BadConnection(), BadConnection())


def test_snapshot_codec_accepts_only_the_current_server_field_set():
    _artifact, service, _connection = _fixture()
    tree = pulse_server_snapshot_to_tree(service.snapshot())

    assert set(tree) == {
        "schema",
        "connection_generation",
        "manifest",
        "clock_hz",
        "geometry_fingerprint",
        "state",
        "prepared_ref",
        "physical_state",
        "physical_prepared_artifact_digest",
        "physical_scan_point_count",
        "physical_scan_cursor",
        "physical_cursor_sample_count",
        "physical_underflow_observed",
        "safe_status_word",
        "safe_clock_enable_words",
    }
    snapshot = pulse_server_snapshot_from_tree(tree)
    assert snapshot.connection_generation == tree["connection_generation"]
    assert snapshot.geometry_fingerprint == tree["geometry_fingerprint"]
    assert snapshot.state == "IDLE"


def test_remote_client_requires_a_physically_distinct_interrupt_connection():
    _artifact, service, connection = _fixture()
    with pytest.raises(ValueError, match="distinct connections"):
        RemotePulseExecutionClient(connection, connection)


def test_remote_client_serializes_concurrent_close_to_one_safe_request(monkeypatch):
    _install_in_process_rpyc_timed(monkeypatch)
    _artifact, service, connection = _fixture()
    interrupt_connection = Connection(service)
    client = RemotePulseExecutionClient(connection, interrupt_connection)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original = interrupt_connection.root.current_interrupt_safe_state

    def blocking_safe(generation):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(1.0)
        return original(generation)

    interrupt_connection.root.current_interrupt_safe_state = blocking_safe
    errors = []

    def close():
        try:
            client.close()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=close)
    second = threading.Thread(target=close)
    first.start()
    assert entered.wait(1.0)
    second.start()
    time.sleep(0.02)
    assert calls == 1
    release.set()
    first.join(1.0)
    second.join(1.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == 1
    assert connection.closed and interrupt_connection.closed


def test_remote_client_terminal_close_revokes_sockets_after_unconfirmed_safe(
    monkeypatch,
):
    _install_in_process_rpyc_timed(monkeypatch)
    _artifact, service, connection = _fixture()
    interrupt_connection = Connection(service)
    client = RemotePulseExecutionClient(connection, interrupt_connection)
    calls = 0

    def fail_safe(_generation):
        nonlocal calls
        calls += 1
        raise RuntimeError("SAFE unconfirmed")

    interrupt_connection.root.current_interrupt_safe_state = fail_safe

    with pytest.raises(RuntimeError, match="SAFE unconfirmed"):
        client.close()
    assert calls == 1
    assert connection.closed and interrupt_connection.closed

    next_connection = Connection(service)
    next_interrupt_connection = Connection(service)
    next_client = RemotePulseExecutionClient(
        next_connection,
        next_interrupt_connection,
    )
    assert next_client.safe_state().safe_readback_confirmed
    assert service.snapshot().safe_readback_confirmed
    next_client.close()


def test_explicit_safe_state_failure_can_retry_on_the_same_live_connection(
    monkeypatch,
):
    _install_in_process_rpyc_timed(monkeypatch)
    _artifact, service, connection = _fixture()
    interrupt_connection = Connection(service)
    client = RemotePulseExecutionClient(connection, interrupt_connection)
    calls = 0
    original = interrupt_connection.root.current_interrupt_safe_state

    def fail_once(generation):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("SAFE temporarily unconfirmed")
        return original(generation)

    interrupt_connection.root.current_interrupt_safe_state = fail_once

    with pytest.raises(RuntimeError, match="temporarily unconfirmed"):
        client.safe_state()
    assert not connection.closed and not interrupt_connection.closed
    assert client.safe_state().safe_readback_confirmed
    assert calls == 2
    client.close()
