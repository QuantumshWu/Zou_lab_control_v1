from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest
from conftest import pulse_backend_completion_for

from zlc_pulse import (
    PulseExecutionForm,
    PulseExecutionService,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_pulse.server import (
    decode_artifact_message,
    decode_prepared_ref_message,
    encode_completion_message,
    encode_prepared_ref_message,
)
from zlc_storage import encode


ROOT = Path(__file__).parents[1]


class Backend:
    def __init__(self) -> None:
        self.prepared = None
        self.safe = True

    def prepare(self, artifact):
        self.prepared = artifact
        self.safe = False

    def fire(self, artifact):
        assert artifact is self.prepared

    def await_completion(self, artifact, timeout):
        assert artifact is self.prepared
        return pulse_backend_completion_for(artifact, transport_id="client-test")

    def safe_state(self):
        self.prepared = None
        self.safe = True

    def request_interrupt(self):
        pass

    def snapshot(self):
        return {"safe": self.safe}


class Root:
    def __init__(self, service: PulseExecutionService) -> None:
        self.service = service

    def current_snapshot(self):
        return encode(self.service.snapshot())

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

    def current_interrupt_safe_state(self, generation):
        return encode(self.service.safe_state_for_generation(generation))


class Connection:
    def __init__(self, service: PulseExecutionService) -> None:
        self.root = Root(service)
        self.closed = False

    def close(self):
        self.closed = True


def _fixture():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    backend = Backend()
    service = PulseExecutionService(document.target, clock_hz=50e6, backend=backend)
    connection = Connection(service)
    return artifact, service, connection


def test_remote_client_runs_one_current_generation_without_legacy_payloads():
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

        def current_interrupt_safe_state(self, generation):
            return encode({"schema": "old-server"})

    class BadConnection:
        root = BadRoot()

        def close(self):
            pass

    with pytest.raises(ValueError, match="unknown field set"):
        RemotePulseExecutionClient(BadConnection(), BadConnection())


def test_remote_client_requires_a_physically_distinct_interrupt_connection():
    _artifact, service, connection = _fixture()
    with pytest.raises(ValueError, match="distinct connections"):
        RemotePulseExecutionClient(connection, connection)


def test_remote_client_serializes_concurrent_close_to_one_safe_request():
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
