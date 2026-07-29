from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import pytest
import zlc_pulse.server as server_module
from conftest import private_pulse_backend_snapshot, pulse_backend_completion_for

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseWireImage,
    PulseExecutionForm,
    PulseExecutionService,
    compile_pulse_artifact,
    decode_artifact_message,
    decode_completion_message,
    decode_prepared_ref_message,
    encode_artifact_message,
    encode_completion_message,
    encode_prepared_ref_message,
    load_pulse_document,
    pulse_server_snapshot_to_tree,
    pulse_target_manifest_from_lanes,
)


ROOT = Path(__file__).parents[1]


def _service_manifest():
    target = load_pulse_document(
        ROOT / "pulses" / "imaging_template.json"
    ).target
    return pulse_target_manifest_from_lanes(target)


class RecordingBackend:
    def __init__(self):
        self.actions = []
        self.prepared = None
        self.done = True
        self.fail_safe = False
        self.state = "IDLE"
        self.scan_points = 0

    def prepare(self, artifact):
        self.actions.append(("prepare", artifact.fingerprint))
        self.prepared = artifact
        self.state = "PREPARED"
        self.scan_points = len(artifact.target_ir.scan_points)

    def fire(self, artifact):
        assert artifact is self.prepared
        self.actions.append(("fire", artifact.fingerprint))
        self.state = "RUNNING"

    def await_completion(self, artifact, timeout):
        assert artifact is self.prepared
        self.actions.append(("await_completion", artifact.fingerprint, timeout))
        if not self.done:
            return None
        self.state = "DONE"
        return pulse_backend_completion_for(artifact)

    def safe_state(self):
        self.actions.append(("safe",))
        if self.fail_safe:
            raise RuntimeError("safe readback failed")
        self.prepared = None
        self.state = "SAFE"

    def request_interrupt(self):
        pass

    def snapshot(self):
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=len(_service_manifest().target.raw_lanes),
            artifact=self.prepared,
            scan_point_count=self.scan_points,
        )


def _artifact(params=None, execution_form=PulseExecutionForm.STATIC_ONCE):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    return compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=execution_form,
        trigger_channels=() if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR else ("ch11",),
        params=params,
    )


def test_server_executes_one_exact_current_artifact_and_returns_schedule_receipt():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="server-generation-1",
    )

    reference = service.prepare(artifact)
    service.fire(reference)
    completion = service.complete(reference, timeout=3.0)

    assert completion.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert completion.hardware_terminal.status_first == completion.hardware_terminal.status_second
    assert reference.artifact_digest == artifact.fingerprint
    assert [action[0] for action in backend.actions] == [
        "prepare",
        "fire",
        "await_completion",
    ]
    snapshot = service.snapshot()
    tree = pulse_server_snapshot_to_tree(snapshot)
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
    assert snapshot.state == "DONE"
    assert snapshot.physical_state == "DONE"


def test_server_messages_are_current_canonical_owner_codecs():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="server-generation-1",
    )
    decoded = decode_artifact_message(encode_artifact_message(artifact))
    reference = service.prepare(decoded)

    assert decoded == artifact
    assert decode_prepared_ref_message(encode_prepared_ref_message(reference)) == reference
    service.fire(reference)
    completion = service.complete(reference, timeout=1.0)
    assert decode_completion_message(encode_completion_message(completion)) == completion


def test_artifact_rpc_message_delegates_to_the_artifact_owner(monkeypatch):
    artifact = _artifact()
    encoded = object()
    decoded = object()
    encode_calls = []
    decode_calls = []

    def owner_encode(value):
        encode_calls.append(value)
        return encoded

    def owner_decode(payload):
        decode_calls.append(payload)
        return decoded

    monkeypatch.setattr(server_module, "encode_compiled_pulse_artifact", owner_encode)
    monkeypatch.setattr(server_module, "decode_compiled_pulse_artifact", owner_decode)

    assert server_module.encode_artifact_message(artifact) is encoded
    assert server_module.decode_artifact_message(b"owner-payload") is decoded
    assert encode_calls == [artifact]
    assert decode_calls == [b"owner-payload"]


def test_completed_receipt_is_replayed_without_reentering_the_backend():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="server-generation-1",
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    first = service.complete(reference, timeout=1.0)
    second = service.complete(reference, timeout=1.0)

    assert second is first
    assert [action[0] for action in backend.actions].count("await_completion") == 1
    with pytest.raises(RuntimeError, match="differs from cached completion"):
        service.complete(
            replace(reference, artifact_digest="0" * 64),
            timeout=1.0,
        )


def test_stale_generation_and_geometry_fail_before_backend_prepare():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="server-generation-1",
    )
    reference = service.prepare(artifact)
    service.safe_state()
    with pytest.raises(RuntimeError, match="state"):
        service.fire(reference)

    wrong_geometry = _artifact(replace(StreamerParams(), bank_size=1024))
    with pytest.raises(ValueError, match="geometry"):
        service.prepare(wrong_geometry)
    assert [action[0] for action in backend.actions] == ["prepare", "safe"]


def test_wire_image_must_equal_the_deterministic_current_ir_packing():
    artifact = _artifact()
    words = list(artifact.wire_image.words)
    address, value = words[-1]
    words[-1] = (address, value ^ 1)
    tampered = CompiledPulseArtifact(
        artifact.source_document_digest,
        artifact.compiler_id,
        artifact.execution_form,
        artifact.target_ir,
        PulseWireImage(
            artifact.wire_image.geometry_fingerprint,
            artifact.wire_image.source_ir_digest,
            tuple(words),
        ),
        artifact.trigger_schedules,
    )
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )

    with pytest.raises(ValueError, match="deterministic TargetIR packing"):
        service.prepare(tampered)
    assert backend.actions == []


def test_timeout_is_not_reported_as_a_completed_schedule():
    artifact = _artifact()
    backend = RecordingBackend()
    backend.done = False
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    with pytest.raises(TimeoutError, match="validated terminal"):
        service.complete(reference, timeout=0.1)
    assert service.snapshot().state == "TIMEOUT"

    with pytest.raises(RuntimeError, match="requires completion or verified safe_state"):
        service.prepare(artifact)
    backend.done = True
    assert service.complete(reference, timeout=0.1).prepared_ref == reference
    assert service.prepare(artifact).artifact_digest == artifact.fingerprint


def test_independent_safe_interrupts_a_blocked_completion_without_waiting_for_its_lock():
    artifact = _artifact()

    class BlockingBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.interrupted = threading.Event()

        def await_completion(self, artifact, timeout):
            self.entered.set()
            self.interrupted.wait(2.0)
            raise RuntimeError("completion interrupted")

        def request_interrupt(self):
            self.interrupted.set()

    backend = BlockingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    reference = service.prepare(artifact)
    service.fire(reference)
    errors = []

    def complete():
        try:
            service.complete(reference, timeout=1.0)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=complete)
    worker.start()
    assert backend.entered.wait(1.0)
    started = time.monotonic()
    service.safe_state()
    elapsed = time.monotonic() - started
    worker.join(1.0)

    assert elapsed < 0.5
    assert not worker.is_alive()
    assert errors and "completion interrupted" in str(errors[0])
    assert service.snapshot().state == "SAFE"
    assert service.snapshot().safe_readback_confirmed


def test_safe_during_artifact_validation_is_a_terminal_admission_fence(monkeypatch):
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    entered = threading.Event()
    release = threading.Event()
    original = service._validate_artifact

    def blocked_validation(value):
        entered.set()
        assert release.wait(1.0)
        original(value)

    monkeypatch.setattr(service, "_validate_artifact", blocked_validation)
    errors = []

    def prepare():
        try:
            service.prepare(artifact)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prepare)
    worker.start()
    assert entered.wait(1.0)
    service.safe_state()
    release.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert errors and "superseded" in str(errors[0])
    assert [action[0] for action in backend.actions] == ["safe"]
    assert service.snapshot().state == "SAFE"


def test_continuous_execution_has_no_false_logical_completion():
    artifact = _artifact(execution_form=PulseExecutionForm.CONTINUOUS_MONITOR)
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    with pytest.raises(RuntimeError, match="no logical completion"):
        service.complete(reference, timeout=1.0)
    assert [action[0] for action in backend.actions] == ["prepare", "fire"]
    assert service.snapshot().state == "RUNNING"


def test_failed_safe_is_never_published_as_safe():
    artifact = _artifact()
    backend = RecordingBackend()
    backend.fail_safe = True
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    service.prepare(artifact)

    with pytest.raises(RuntimeError, match="safe readback failed"):
        service.safe_state()

    assert service.snapshot().state == "SAFE_FAILED"
    assert not service.snapshot().safe_readback_confirmed

    failed_generation = service.connection_generation
    backend.fail_safe = False
    service.safe_state()
    recovered = service.snapshot()
    assert recovered.state == "SAFE"
    assert recovered.safe_readback_confirmed
    assert service.renew_connection_generation() != failed_generation


def test_new_connection_generation_permanently_invalidates_old_prepared_refs():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="server-generation-1",
    )
    old_reference = service.prepare(artifact)

    with pytest.raises(RuntimeError, match="cannot renew"):
        service.renew_connection_generation()

    service.safe_state()
    assert service.renew_connection_generation() != "server-generation-1"
    new_reference = service.prepare(artifact)
    assert new_reference != old_reference

    with pytest.raises(RuntimeError, match="stale"):
        service.fire(old_reference)


def test_interrupt_receipt_never_leaks_a_new_connection_generation(monkeypatch):
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        _service_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="old-generation",
    )
    service.prepare(artifact)
    original = service._safe_state

    def disconnect_and_replace_owner(*, expected_generation):
        receipt = original(expected_generation=expected_generation)
        service.renew_connection_generation()
        return receipt

    monkeypatch.setattr(service, "_safe_state", disconnect_and_replace_owner)

    receipt = service.safe_state_for_generation("old-generation")
    assert receipt.connection_generation == "old-generation"
    assert receipt.state == "SAFE"
    assert receipt.safe_readback_confirmed
    assert service.connection_generation != "old-generation"
