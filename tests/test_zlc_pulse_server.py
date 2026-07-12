from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

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
)


ROOT = Path(__file__).parents[1]


class RecordingBackend:
    def __init__(self):
        self.actions = []
        self.prepared = None
        self.done = True
        self.fail_safe = False

    def prepare(self, artifact):
        self.actions.append(("prepare", artifact.fingerprint))
        self.prepared = artifact

    def fire(self, artifact):
        assert artifact is self.prepared
        self.actions.append(("fire", artifact.fingerprint))

    def wait_done(self, artifact, timeout):
        assert artifact is self.prepared
        self.actions.append(("wait_done", artifact.fingerprint, timeout))
        return self.done

    def safe_state(self):
        self.actions.append(("safe",))
        if self.fail_safe:
            raise RuntimeError("safe readback failed")
        self.prepared = None

    def snapshot(self):
        return {"prepared": self.prepared is not None}


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
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
        connection_generation="server-generation-1",
    )

    reference = service.prepare(artifact)
    service.fire(reference)
    completion = service.complete(reference, timeout=3.0)

    assert completion.logical_done
    assert completion.completed_schedule_trigger_counts == (("ch11", 3),)
    assert reference.artifact_digest == artifact.fingerprint
    assert [action[0] for action in backend.actions] == ["prepare", "fire", "wait_done"]
    assert service.snapshot()["state"] == "DONE"


def test_server_messages_are_current_canonical_owner_codecs():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
        connection_generation="server-generation-1",
    )
    decoded = decode_artifact_message(encode_artifact_message(artifact))
    reference = service.prepare(decoded)

    assert decoded == artifact
    assert decode_prepared_ref_message(encode_prepared_ref_message(reference)) == reference
    service.fire(reference)
    completion = service.complete(reference, timeout=1.0)
    assert decode_completion_message(encode_completion_message(completion)) == completion


def test_stale_generation_and_geometry_fail_before_backend_prepare():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
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
        artifact.compiler_version,
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
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
    )

    with pytest.raises(ValueError, match="deterministic TargetIR packing"):
        service.prepare(tampered)
    assert backend.actions == []


def test_timeout_is_not_reported_as_a_completed_schedule():
    artifact = _artifact()
    backend = RecordingBackend()
    backend.done = False
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    completion = service.complete(reference, timeout=0.1)

    assert not completion.logical_done
    assert completion.completed_schedule_trigger_counts == ()
    assert service.snapshot()["state"] == "TIMEOUT"

    with pytest.raises(RuntimeError, match="requires completion or verified safe_state"):
        service.prepare(artifact)
    service.safe_state()
    assert service.prepare(artifact).artifact_digest == artifact.fingerprint


def test_continuous_execution_has_no_false_logical_completion():
    artifact = _artifact(execution_form=PulseExecutionForm.CONTINUOUS_MONITOR)
    backend = RecordingBackend()
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    with pytest.raises(RuntimeError, match="no logical completion"):
        service.complete(reference, timeout=1.0)
    assert [action[0] for action in backend.actions] == ["prepare", "fire"]
    assert service.snapshot()["state"] == "RUNNING"


def test_failed_safe_is_never_published_as_safe():
    artifact = _artifact()
    backend = RecordingBackend()
    backend.fail_safe = True
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
    )
    service.prepare(artifact)

    with pytest.raises(RuntimeError, match="safe readback failed"):
        service.safe_state()

    assert service.snapshot()["state"] == "SAFE_FAILED"


def test_new_connection_generation_permanently_invalidates_old_prepared_refs():
    artifact = _artifact()
    backend = RecordingBackend()
    service = PulseExecutionService(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json").target,
        clock_hz=50e6,
        backend=backend,
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
