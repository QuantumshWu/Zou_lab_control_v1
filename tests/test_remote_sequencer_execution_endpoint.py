"""Current target-owned remote pulse endpoint and interrupt fencing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import pytest

from conftest import private_pulse_backend_snapshot, pulse_backend_completion_for

from fpga.pulse_streamer.host.image import StreamerParams

from zlc_neutral_atom.devices.sequencer.remote_pulse import (
    RemotePulseExecutionEndpoint,
)
from zlc_neutral_atom.runtime.ports import (
    DeviceBroker,
    SafetyOperation,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_neutral_atom.devices.sequencer.port import (
    CompletePulseCommand,
    ContinuousPulseExecutionRequest,
    FinitePulseExecutionRequest,
    FirePulseCommand,
    PreparePulseCommand,
    PulseTerminalEvidenceKind,
    SequencerCapabilitySnapshot,
)
from zlc_pulse import (
    PulseCompletion,
    PulseExecutionForm,
    PulseExecutionService,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    freeze_scan_table,
    load_pulse_document,
    pulse_server_snapshot_from_tree,
    pulse_target_manifest_from_lanes,
)
from zlc_pulse.server import (
    decode_artifact_message,
    decode_prepared_ref_message,
    encode_completion_message,
    encode_continuous_failure_message,
    encode_prepared_ref_message,
    pulse_server_snapshot_to_tree,
)
from zlc_storage import decode, encode


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


class Backend:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.prepared = None
        self.safe = True
        self.completion = None
        self.continuous_failure = None
        self.state = "IDLE"
        self.scan_points = 0

    def prepare(self, artifact):
        self.actions.append("prepare")
        self.prepared = artifact
        self.safe = False
        self.state = "PREPARED"
        self.scan_points = len(artifact.target_ir.scan_points)

    def fire(self, artifact):
        assert artifact is self.prepared
        self.actions.append("fire")
        self.state = "RUNNING"

    def await_completion(self, artifact, timeout):
        assert artifact is self.prepared
        self.actions.append("wait")
        self.completion = pulse_backend_completion_for(
            artifact,
            transport_id="remote-test",
        )
        self.state = "DONE"
        return self.completion

    def wait_continuous_failure(self, artifact, timeout):
        assert artifact is self.prepared
        time.sleep(min(float(timeout), 0.001))
        return self.continuous_failure

    def safe_state(self):
        self.actions.append("safe")
        self.prepared = None
        self.safe = True
        self.state = "SAFE"

    def request_interrupt(self):
        return None

    def snapshot(self):
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=_raw_lane_count(),
            artifact=self.prepared,
            scan_point_count=self.scan_points,
        )


class Root:
    def __init__(self, service):
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

    def current_interrupt_safe_state(self):
        return encode(
            pulse_server_snapshot_to_tree(
                self.service.safe_state_for_generation(
                    self.service.connection_generation
                )
            )
        )


class Connection:
    def __init__(self, service):
        self.root = Root(service)
        self.closed = False

    def close(self):
        self.closed = True


class InProcessRemotePulseExecutionClient(RemotePulseExecutionClient):
    """Current client semantics over a synchronous in-process RPC double."""

    def _safe_state_owned(self, logical_timeout):
        self._require_open()
        snapshot = pulse_server_snapshot_from_tree(
            decode(
                bytes(
                    self._interrupt_root.current_interrupt_safe_state()
                )
            )
        )
        if snapshot.connection_generation != self.connection_generation:
            raise RuntimeError("interrupt safe_state returned another connection generation")
        if (
            snapshot.state != "SAFE"
            or snapshot.prepared_ref is not None
            or not snapshot.safe_readback_confirmed
        ):
            raise RuntimeError("pulse server acknowledged safe_state without publishing SAFE")
        return snapshot


def _raw_lane_count() -> int:
    return len(load_pulse_document(IMAGING_TEMPLATE).target.raw_lanes)


def _bound_remote(client, endpoint, *, suffix="main"):
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: PhysicalDeviceIdentity(
            stable_device_identity=f"installation-endpoint:test-fpga-{suffix}",
            evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        )
    )
    binding = None

    def current_binding():
        assert binding is not None
        return binding

    binding = broker.bind(
        key=ResourceKey.parse(f"device/sequencer/remote-{suffix}"),
        identity=identity,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(), command
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(), command
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    capability = broker.verify_capability(binding).snapshot
    assert isinstance(capability, SequencerCapabilitySnapshot)
    return broker, binding, capability


def _commands(request, capability, *, session_id, run_id):
    artifact = request.artifact
    return (
        PreparePulseCommand(
            session_id,
            run_id,
            request,
            capability.capability_fingerprint,
            5.0,
        ),
        FirePulseCommand(session_id, artifact.fingerprint),
        CompletePulseCommand(session_id, artifact.fingerprint, 5.0),
    )


def test_remote_current_endpoint_runs_exact_artifact_and_closes_safe() -> None:
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
    control = Connection(service)
    interrupt = Connection(service)
    client = InProcessRemotePulseExecutionClient(
        control,
        interrupt,
        transport_timeout_seconds=10.0,
    )
    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label="test-fpga",
        max_blocking_call_seconds=5.0,
    )
    broker, binding, capability = _bound_remote(client, endpoint)
    request = FinitePulseExecutionRequest(document, artifact)
    prepare, fire, complete = _commands(
        request,
        capability,
        session_id="remote-session",
        run_id="remote-run",
    )

    endpoint.execute_command(binding, prepare)
    endpoint.execute_command(binding, fire)
    terminal = endpoint.execute_command(binding, complete)
    closed = endpoint.close_session(
        binding,
        SessionCloseCommand("remote-session", 5.0),
    )

    assert isinstance(terminal.receipt, PulseCompletion)
    assert terminal.evidence_kind is PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS
    assert terminal.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert terminal.receipt.hardware_terminal.transport_id == "remote-test"
    assert terminal.receipt.hardware_terminal == backend.completion.hardware_terminal
    assert terminal.receipt.post_terminal_tail == backend.completion.post_terminal_tail
    assert terminal.artifact_digest == artifact.fingerprint
    assert closed.is_terminal
    assert service.snapshot().state == "SAFE"
    assert service.snapshot().safe_readback_confirmed
    assert backend.actions == ["prepare", "fire", "wait", "safe"]

    broker.shutdown()
    client.close()
    assert control.closed and interrupt.closed


def test_remote_continuous_scan_progress_requires_a_real_cursor_sample() -> None:
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    document = replace(document, scan_table=table)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    )

    class ProgressBackend(Backend):
        def __init__(self) -> None:
            super().__init__()
            self.fired = False
            self.cursor = 0
            self.cursor_sample_count = 0

        def fire(self, current):
            super().fire(current)
            self.fired = True

        def safe_state(self):
            super().safe_state()
            self.fired = False

        def snapshot(self):
            return private_pulse_backend_snapshot(
                state=self.state,
                raw_lane_count=_raw_lane_count(),
                artifact=self.prepared,
                scan_point_count=self.scan_points,
                cursor=self.cursor,
                cursor_sample_count=self.cursor_sample_count,
            )

    backend = ProgressBackend()
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    control = Connection(service)
    interrupt = Connection(service)
    client = InProcessRemotePulseExecutionClient(
        control,
        interrupt,
        transport_timeout_seconds=10.0,
    )
    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label="test-progress-fpga",
        max_blocking_call_seconds=5.0,
    )
    broker, binding, capability = _bound_remote(
        client,
        endpoint,
        suffix="progress",
    )
    request = ContinuousPulseExecutionRequest(document, artifact)
    prepare = PreparePulseCommand(
        "progress-session",
        "progress-run",
        request,
        capability.capability_fingerprint,
        5.0,
    )
    fire = FirePulseCommand("progress-session", artifact.fingerprint)
    endpoint.execute_command(binding, prepare)
    endpoint.execute_command(binding, fire)

    unavailable = endpoint.observe_scan_progress(
        binding,
        "progress-session",
        "progress-run",
        artifact.fingerprint,
        2,
    )
    assert not unavailable.available
    assert "not sampled" in unavailable.unavailable_reason

    backend.cursor = 1
    backend.cursor_sample_count = 1
    progress = endpoint.observe_scan_progress(
        binding,
        "progress-session",
        "progress-run",
        artifact.fingerprint,
        2,
    )
    assert progress.available
    assert progress.current_point_index == 1

    backend.continuous_failure = "forced deployed observer failure"
    failure = endpoint.wait_continuous_failure(
        binding,
        "progress-session",
        "progress-run",
        artifact.fingerprint,
        0.1,
    )
    assert failure == "forced deployed observer failure"
    assert service.snapshot().state == "FAILED"

    endpoint.close_session(
        binding,
        SessionCloseCommand("progress-session", 5.0),
    )
    broker.shutdown()
    client.close()


def test_interrupt_fences_a_provisional_remote_prepare_before_it_can_fire() -> None:
    document = load_pulse_document(IMAGING_TEMPLATE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )

    class BlockingBeforeServiceRoot(Root):
        def __init__(self, service):
            super().__init__(service)
            self.entered = threading.Event()
            self.release = threading.Event()

        def current_prepare(self, payload):
            self.entered.set()
            assert self.release.wait(5.0)
            return super().current_prepare(payload)

    backend = Backend()
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    control = Connection(service)
    blocking_root = BlockingBeforeServiceRoot(service)
    control.root = blocking_root
    interrupt = Connection(service)
    client = InProcessRemotePulseExecutionClient(
        control,
        interrupt,
        transport_timeout_seconds=10.0,
    )
    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label="test-fpga-race",
        max_blocking_call_seconds=5.0,
    )
    broker, binding, capability = _bound_remote(
        client,
        endpoint,
        suffix="race",
    )
    request = FinitePulseExecutionRequest(document, artifact)
    command, _fire, _complete = _commands(
        request,
        capability,
        session_id="race-session",
        run_id="race-run",
    )
    errors = []

    def prepare():
        try:
            endpoint.execute_command(binding, command)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prepare)
    worker.start()
    assert blocking_root.entered.wait(1.0), errors
    endpoint.interrupt()

    retry_command = PreparePulseCommand(
        "retry-session",
        "retry-run",
        request,
        capability.capability_fingerprint,
        5.0,
    )
    with pytest.raises(RuntimeError, match="physical operation in flight"):
        endpoint.execute_command(binding, retry_command)
    with pytest.raises(TimeoutError, match="did not join"):
        endpoint.close_session(
            binding,
            SessionCloseCommand(command.session_id, 0.01),
        )
    assert worker.is_alive()

    blocking_root.release.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "superseded by interrupt" in str(errors[0])
    # The late prepare crossed the first interrupt, so sealing that failed
    # operation requires one later physical SAFE.  A joined close must reuse
    # that exact SAFE receipt rather than issue another physical transition.
    assert backend.actions == ["safe", "prepare", "safe"]
    assert service.snapshot().state == "SAFE"

    # Reuse remains fenced until a joined, terminally acknowledged close.
    closed = endpoint.close_session(
        binding,
        SessionCloseCommand(command.session_id, 5.0),
    )
    assert closed.is_terminal
    assert backend.actions == ["safe", "prepare", "safe"]
    acknowledgement = endpoint.execute_command(binding, retry_command)
    assert acknowledgement.session_id == retry_command.session_id
    retry_closed = endpoint.close_session(
        binding,
        SessionCloseCommand(retry_command.session_id, 5.0),
    )
    assert retry_closed.is_terminal

    broker.shutdown()
    client.close()
